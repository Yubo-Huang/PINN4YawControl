import os
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process
from Networks.seq2seq_ConvLSTM import EncoderDecoderConvLSTM
from WindForeDataset import WindDataset 
from utilize import EarlyStopping

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')
parser.add_argument('--n_step_past', type=int, default=20, help='number of past frames')
parser.add_argument('--n_step_ahead', type=int, default=20, help='number of ahead frames')
parser.add_argument('--train_name', type=str, default='adam_huber', help='name of training, such as optimizer_lossfun')

opt = parser.parse_args()

log_file_path = 'WindFore_logs/' + opt.train_name
if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
if not os.path.isdir(log_file_path):
    raise Exception('%s is not a dir' % log_file_path)
writer = SummaryWriter(log_file_path)

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

def create_video(x, y_hat, y):
    # predictions with input for illustration purposes
    preds = torch.cat([x.cpu(), y_hat.cpu()], dim=1)[0]
    preds_norm = torch.sqrt(torch.sum(torch.pow(preds, 2), dim=1))
    # entire input and ground truth
    y_plot = torch.cat([x.cpu(), y.cpu()], dim=1)[0]
    y_plot_norm = torch.sqrt(torch.sum(torch.pow(y_plot, 2), dim=1))

    # error (l2 norm) plot between pred and ground truth
    y_hat_norm = torch.sqrt(torch.sum(torch.pow(y_hat[0], 2), dim=1))
    y_norm = torch.sqrt(torch.sum(torch.pow(y[0], 2), dim=1))
    difference = (torch.pow(y_hat_norm - y_norm, 2)).detach().cpu()
    zeros = torch.zeros(difference.shape)
    difference_plot = torch.cat([zeros.cpu(), difference.cpu()], dim=0)

    # concat all images
    final_image = torch.cat([preds_norm, y_plot_norm, difference_plot], dim=0)

    # make them into a single grid image file
    grid = torchvision.utils.make_grid(final_image, nrow=opt.n_steps_past + opt.n_steps_ahead)

    return grid

def train_epoch(model, loader, optimizer, epoch, n_epochs, n_step_ahead, print_freq=1000):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, batch_samples in enumerate(loader):
        # Create vaiables
        [input, label] = batch_samples
        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()

        # compute output
        pred = model(input, n_step_ahead)
        # loss = F.mse_loss(pred, label)
        loss = F.huber_loss(pred, label, delta=0.5)
        # measure accuracy and record loss
        batch_size = input.size(0)
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
            writer.add_scalar('Batch time', batch_time.avg, epoch * len(loader)+batch_idx+1)
            
    # Creat the video of wind speed after each epoch
    # final_image = create_video(input, pred, label)
    # Return summary statistics
    return batch_time.avg, losses.avg

def test_epoch(model, loader, n_step_ahead, epoch):
    pred_errors = AverageMeter()
    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(loader):
            [input, label] = batch_samples
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            # compute output
            pred = model(input, n_step_ahead)
            # loss = F.mse_loss(pred, label)
            error = F.huber_loss(pred, label, delta=0.5)
            batch_size = input.size(0)
            pred_errors.update(error.item(), batch_size)
    writer.add_scalar('Validation error', error, epoch)
           
    # Return summary statistics
    return pred_errors.avg

def train(model, train_set, valid_set, save, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=32)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False,
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
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=opt.lr)
    n_epochs = opt.epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)
    
    early_stopping = EarlyStopping(patience=3)
    # Train model
    n_step_ahead = opt.n_step_ahead
    for epoch in range(n_epochs):
        train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            n_step_ahead=n_step_ahead
        )
        scheduler.step()
        
        # save the model every epoch
        torch.save(model.state_dict(), os.path.join(save, 'model_epoch{}.pth'.format(epoch)))
        
        # test model
        error = test_epoch(model, loader=valid_loader, n_step_ahead=n_step_ahead, epoch=epoch)
        
        # Check if validation loss has improved, and perform early stopping if necessary
        early_stopping(error, epoch)
        # If early stopping criterion is met, break out of the training loop
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch} as the valid error has not been decreased within 3 epochs')
            break
    print(f'the model with the best performact (lowest validation error) is trained at Epoch {early_stopping.best_epoch}')

def main(train_data_dir, valid_data_dir, save_dir, seed=None, restore=False):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # Models
    model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=3)
    print(model)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)
    
    train_set = WindDataset(train_data_dir)
    valid_set = WindDataset(valid_data_dir)

    # Make save directory
    save_dir = os.path.join(save_dir, opt.train_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(save_dir):
        raise Exception('%s is not a dir' % save_dir)
    if restore:
        model_files = os.listdir(save_dir)
        model_files = [f for f in model_files if f.endswith('.pth')]
        latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
        latest_model_file = os.path.join(save_dir, latest_model_file)
        model_state_dict  = torch.load(latest_model_file)
        model.load_state_dict(model_state_dict)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, save=save_dir, seed=seed)
    print('Done!')

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
    train_data_dir = './WindFore_train_dataset'
    valid_data_dir = './WindFore_valid_dataset'
    save_dir = './WindFore_trained_model'
    main(train_data_dir, valid_data_dir, save_dir)

