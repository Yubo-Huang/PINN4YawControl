import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Networks.DenseNet import DenseNet
from DeepALMDataset import WindDataset 
from utilize import EarlyStopping

writer = SummaryWriter('DeepALMlogs/adam_mse')

def ALMLossFun(equivalent_wind, 
               body_velocity, 
               normal_vec, 
               tangent_vec, 
               normal_coef,
               tangent_coef,
               moment_coef,
               normal_force,
               tangent_force, 
               moment_torque):
        area = torch.tensor([159.89, 178.40, 168.68, 149.28, 128.99, 108.96, 77.68])
        chord = torch.tensor([8.75, 9.76, 9.23, 8.17, 7.06, 5.96, 4.25])
        normal_norm = 0.5*torch.tensor([1e3, 1e4, 1e4, 1e4, 1e5, 1e5, 1e5])
        if torch.cuda.is_available():
            area = area.cuda()
            chord = chord.cuda()
            normal_norm = normal_norm.cuda()
        tensor_shape = body_velocity.shape
        
        equal_wind_expand = torch.unsqueeze(torch.unsqueeze(equivalent_wind, dim=1), dim=2)
        
        relative_wind = torch.sub(equal_wind_expand, body_velocity)
        normal_velocity = torch.sum(torch.mul(relative_wind, normal_vec), dim=3)
        tangent_velocity = torch.sum(torch.mul(relative_wind, tangent_vec), dim=3)
        relative_wind_squared = torch.add(torch.pow(normal_velocity, 2), torch.pow(tangent_velocity, 2))
        
        normal_para = torch.mul(torch.mul(0.5, normal_coef), area) 
        equal_normal_force = torch.mul(normal_para, relative_wind_squared)
        tangent_para = torch.mul(torch.mul(0.5, tangent_coef), area)
        equal_tangent_force = torch.mul(tangent_para, relative_wind_squared)
        moment_para = torch.mul(torch.mul(torch.mul(0.5, moment_coef), chord), area) 
        equal_moment_torque = torch.mul(moment_para, relative_wind_squared)  
        
        
        # normal_force_loss = F.mse_loss(torch.div(equal_normal_force, normal_norm), torch.div(normal_force, normal_norm))
        # tangent_force_loss = F.mse_loss(torch.div(equal_tangent_force, 1e3), torch.div(tangent_force, 1e3))
        # moment_torque_loss = F.mse_loss(torch.div(equal_moment_torque, 2e4), torch.div(moment_torque, 2e4))
        # normal_force_loss = F.huber_loss(torch.div(equal_normal_force, normal_norm), torch.div(normal_force, normal_norm), delta=0.5)
        # tangent_force_loss = F.huber_loss(torch.div(equal_tangent_force, 1e3), torch.div(tangent_force, 1e3), delta=0.5)
        # moment_torque_loss = F.huber_loss(torch.div(equal_moment_torque, 2e4), torch.div(moment_torque, 2e4), delta=0.5)
        normal_force_loss = F.mse_loss(equal_normal_force, normal_force)
        tangent_force_loss = 10 * F.mse_loss(equal_tangent_force, tangent_force)
        moment_torque_loss = F.mse_loss(equal_moment_torque, moment_torque)
        ALMLoss = torch.add(torch.add(normal_force_loss, tangent_force_loss), moment_torque_loss)
        
        return normal_force_loss, tangent_force_loss, moment_torque_loss, ALMLoss

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1000):
    batch_time = AverageMeter()
    n_losses = AverageMeter()
    t_losses = AverageMeter()
    m_losses = AverageMeter()
    ref_losses = AverageMeter()
    losses = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, batch_samples in enumerate(loader):
        # Create vaiables
        [body_velocity, 
         normal_vector, 
         tangent_vector, 
         normal_coef, 
         tangent_coef, 
         moment_coef, 
         normal_force, 
         tangent_force,
         moment_torque, 
         uniform_wind, 
         wind_blade_info] = batch_samples
        if torch.cuda.is_available():
            body_velocity = body_velocity.cuda()
            normal_vector = normal_vector.cuda()
            tangent_vector = tangent_vector.cuda()
            normal_coef   = normal_coef.cuda()
            tangent_coef = tangent_coef.cuda()
            moment_coef = moment_coef.cuda()
            normal_force = normal_force.cuda()
            tangent_force = tangent_force.cuda()
            moment_torque = moment_torque.cuda()
            uniform_wind = uniform_wind.cuda()
            wind_blade_info = wind_blade_info.cuda()

        # compute output
        equal_wind = model(wind_blade_info)
        ref_loss = F.mse_loss(equal_wind, uniform_wind)
        n_loss, t_loss, m_loss, alm_loss = ALMLossFun(equal_wind, 
                                                      body_velocity, 
                                                      normal_vector, 
                                                      tangent_vector,
                                                      normal_coef,
                                                      tangent_coef,
                                                      moment_coef,
                                                      normal_force,
                                                      tangent_force,
                                                      moment_torque)
        loss = 1e7*ref_loss + alm_loss
        # measure accuracy and record loss
        batch_size = uniform_wind.size(0)
        n_losses.update(n_loss.item(), batch_size)
        t_losses.update(t_loss.item(), batch_size)
        m_losses.update(m_loss.item(), batch_size)
        ref_losses.update(ref_loss.item(), batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg)
            ])
            print(res)
            # Show loss on the tensorboard
            writer.add_scalar('Total loss', losses.avg, epoch * len(loader)+batch_idx+1)
            writer.add_scalar('Reference loss', ref_losses.avg, epoch * len(loader)+batch_idx+1)
            writer.add_scalar('Normal force loss', n_losses.avg, epoch * len(loader)+batch_idx+1)
            writer.add_scalar('Tangent force loss', t_losses.avg, epoch * len(loader)+batch_idx+1)
            writer.add_scalar('Moment torque', m_losses.avg, epoch * len(loader)+batch_idx+1)
            
            # losses.reset()
            # ref_losses.reset()
            # n_losses.reset()
            # t_losses.reset()
            # m_losses.reset()
    # Return summary statistics
    return batch_time.avg, losses.avg

def test_epoch(model, loader, epoch):
    n_errors = AverageMeter()
    t_errors = AverageMeter()
    m_errors = AverageMeter()
    ref_errors = AverageMeter()
    errors = AverageMeter()

    # Model on train mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(loader):
            # Create vaiables
            [body_velocity, 
            normal_vector, 
            tangent_vector, 
            normal_coef, 
            tangent_coef, 
            moment_coef, 
            normal_force, 
            tangent_force,
            moment_torque, 
            uniform_wind, 
            wind_blade_info] = batch_samples
            if torch.cuda.is_available():
                body_velocity = body_velocity.cuda()
                normal_vector = normal_vector.cuda()
                tangent_vector = tangent_vector.cuda()
                normal_coef   = normal_coef.cuda()
                tangent_coef = tangent_coef.cuda()
                moment_coef = moment_coef.cuda()
                normal_force = normal_force.cuda()
                tangent_force = tangent_force.cuda()
                moment_torque = moment_torque.cuda()
                uniform_wind = uniform_wind.cuda()
                wind_blade_info = wind_blade_info.cuda()

            # compute output
            equal_wind = model(wind_blade_info)
            ref_error = F.mse_loss(equal_wind, uniform_wind)
            n_error, t_error, m_error, alm_error = ALMLossFun(equal_wind, 
                                                        body_velocity, 
                                                        normal_vector, 
                                                        tangent_vector,
                                                        normal_coef,
                                                        tangent_coef,
                                                        moment_coef,
                                                        normal_force,
                                                        tangent_force,
                                                        moment_torque)
            error = 1e7*ref_error + alm_error
            # measure accuracy and record loss
            batch_size = uniform_wind.size(0)
            n_errors.update(n_error.item(), batch_size)
            t_errors.update(t_error.item(), batch_size)
            m_errors.update(m_error.item(), batch_size)
            ref_errors.update(ref_error.item(), batch_size)
            errors.update(error.item(), batch_size)
            
        writer.add_scalar('Total errors', errors.avg, epoch)
        writer.add_scalar('Reference errors', ref_errors.avg, epoch)
        writer.add_scalar('Normal force errors', n_errors.avg, epoch)
        writer.add_scalar('Tangent force errors', t_errors.avg, epoch)
        writer.add_scalar('Moment errors', m_errors.avg, epoch)
    # Return summary statistics
    return errors.avg

def train(model, train_set, valid_set, save, n_epochs=300,
          batch_size=64, lr=1e-3, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=32)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=32)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    # optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    early_stopping = EarlyStopping(patience=5)
    # Train model
    for epoch in range(n_epochs):
        train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        scheduler.step()
        
        # save the model every epoch
        torch.save(model.state_dict(), os.path.join(save, 'model_epoch{}.pth'.format(epoch)))
        # test model
        error = test_epoch(model, loader=valid_loader, epoch=epoch)
        # Check if validation loss has improved, and perform early stopping if necessary
        early_stopping(error, epoch)
        # If early stopping criterion is met, break out of the training loop
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch} as the valid error has not been decreased within 3 epochs')
            break
    print(f'the model with the best performact (lowest validation error) is trained at Epoch {early_stopping.best_epoch}')


def main(train_data_dir, valid_data_dir, save_dir, depth=100, growth_rate=12, efficient=True,
         n_epochs=300, batch_size=512, seed=None):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Models
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate*2,
        num_output=3,
        small_inputs=False,
        efficient=efficient,
    )
    print(model)
    train_set = WindDataset(train_data_dir)
    valid_set = WindDataset(valid_data_dir)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Make save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(save_dir):
        raise Exception('%s is not a dir' % save_dir)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, save=save_dir,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.

Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python demo.py --efficient False --data <path_to_data_dir> --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

if __name__ == '__main__':
    set_seed()
    train_data_dir = './DeepALM_train_dataset'
    valid_data_dir = './DeepALM_test_dataset'
    save_dir = './DeepALM_trained_model/adam_mse'
    main(train_data_dir, valid_data_dir, save_dir)