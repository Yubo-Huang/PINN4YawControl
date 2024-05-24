import os
import h5py
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from Networks.DenseNet import DenseNet
from DeepALMTestDataset import WindDataset 

def ALMError(equivalent_wind, 
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
        if torch.cuda.is_available():
            area = area.cuda()
            chord = chord.cuda()
        
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
        
        normal_force_error = torch.abs(torch.sub(equal_normal_force, normal_force))
        normal_force_error = torch.mean(normal_force_error, dim=(1, 2))
        tangent_force_error = torch.abs(torch.sub(equal_tangent_force, tangent_force))
        tangent_force_error = torch.mean(tangent_force_error, dim=(1, 2))
        moment_torque_error = torch.abs(torch.sub(equal_moment_torque, moment_torque))
        moment_torque_error = torch.mean(moment_torque_error, dim=(1, 2))
        
        return normal_force_error, tangent_force_error, moment_torque_error


def test_epoch(model, loader, save_dir, print_freq=1, is_test=True):

    # Model on eval mode
    model.eval()
    
    r_errors, n_errors, t_errors, m_errors = [], [], [], []
    alm_r_errors, alm_n_errors, alm_t_errors, alm_m_errors = [], [], [], []
    hub_r_errors, hub_n_errors, hub_t_errors, hub_m_errors = [], [], [], []
    mean_r_errors, mean_n_errors, mean_t_errors, mean_m_errors = [], [], [], []
    deep_wind_info, alm_wind_info, hub_wind_info, mean_wind_info = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(loader):
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
            wind_blade_info,
            alm_wind,
            hub_wind,
            mean_wind] = batch_samples
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
                alm_wind  = alm_wind.cuda()
                hub_wind  = hub_wind.cuda()
                mean_wind = mean_wind.cuda()

            # compute output
            equal_wind = model(wind_blade_info)
            reference_error = torch.sqrt(torch.sum(torch.pow((equal_wind - uniform_wind), 2), dim=-1))
            alm_reference_error = torch.sqrt(torch.sum(torch.pow((alm_wind - uniform_wind), 2), dim=-1))
            hub_reference_error = torch.sqrt(torch.sum(torch.pow((hub_wind - uniform_wind), 2), dim=-1))
            mean_reference_error = torch.sqrt(torch.sum(torch.pow((mean_wind - uniform_wind), 2), dim=-1))
            normal_force_error, tangent_force_error, moment_torque_error = ALMError(equal_wind, 
                                                                                    body_velocity, 
                                                                                    normal_vector, 
                                                                                    tangent_vector, 
                                                                                    normal_coef,
                                                                                    tangent_coef,
                                                                                    moment_coef,
                                                                                    normal_force,
                                                                                    tangent_force,
                                                                                    moment_torque)
            alm_normal_force_error, alm_tangent_force_error, alm_moment_torque_error = ALMError(alm_wind, 
                                                                                    body_velocity, 
                                                                                    normal_vector, 
                                                                                    tangent_vector, 
                                                                                    normal_coef,
                                                                                    tangent_coef,
                                                                                    moment_coef,
                                                                                    normal_force,
                                                                                    tangent_force,
                                                                                    moment_torque)
            hub_normal_force_error, hub_tangent_force_error, hub_moment_torque_error = ALMError(hub_wind, 
                                                                                    body_velocity, 
                                                                                    normal_vector, 
                                                                                    tangent_vector, 
                                                                                    normal_coef,
                                                                                    tangent_coef,
                                                                                    moment_coef,
                                                                                    normal_force,
                                                                                    tangent_force,
                                                                                    moment_torque)
            mean_normal_force_error, mean_tangent_force_error, mean_moment_torque_error = ALMError(mean_wind, 
                                                                                    body_velocity, 
                                                                                    normal_vector, 
                                                                                    tangent_vector, 
                                                                                    normal_coef,
                                                                                    tangent_coef,
                                                                                    moment_coef,
                                                                                    normal_force,
                                                                                    tangent_force,
                                                                                    moment_torque)
            r_errors.append(reference_error.cpu().numpy())
            n_errors.append(normal_force_error.cpu().numpy())
            t_errors.append(tangent_force_error.cpu().numpy())
            m_errors.append(moment_torque_error.cpu().numpy())
            
            alm_r_errors.append(alm_reference_error.cpu().numpy())
            alm_n_errors.append(alm_normal_force_error.cpu().numpy())
            alm_t_errors.append(alm_tangent_force_error.cpu().numpy())
            alm_m_errors.append(alm_moment_torque_error.cpu().numpy())
            
            hub_r_errors.append(hub_reference_error.cpu().numpy())
            hub_n_errors.append(hub_normal_force_error.cpu().numpy())
            hub_t_errors.append(hub_tangent_force_error.cpu().numpy())
            hub_m_errors.append(hub_moment_torque_error.cpu().numpy())

            mean_r_errors.append(mean_reference_error.cpu().numpy())
            mean_n_errors.append(mean_normal_force_error.cpu().numpy())
            mean_t_errors.append(mean_tangent_force_error.cpu().numpy())
            mean_m_errors.append(mean_moment_torque_error.cpu().numpy())
            
            deep_wind_info.append(equal_wind.cpu().numpy())
            alm_wind_info.append(alm_wind.cpu().numpy())
            hub_wind_info.append(hub_wind.cpu().numpy())
            mean_wind_info.append(mean_wind.cpu().numpy())
            print(f'finalized the {batch_idx}-th batch')
            
            # print(torch.mean(normal_force_error))
            # print(torch.mean(tangent_force_error))
            # print(torch.mean(moment_torque_error))
            # print(torch.mean(reference_error))
            # print(torch.mean(alm_normal_force_error))
            # print(torch.mean(alm_tangent_force_error))
            # print(torch.mean(alm_moment_torque_error))
            # print(torch.mean(alm_reference_error))
            # print(torch.mean(hub_normal_force_error))
            # print(torch.mean(hub_tangent_force_error))
            # print(torch.mean(hub_moment_torque_error))
            # print(torch.mean(hub_reference_error))
            # print(torch.mean(mean_normal_force_error))
            # print(torch.mean(mean_tangent_force_error))
            # print(torch.mean(mean_moment_torque_error))
            # print(torch.mean(mean_reference_error))
            # print('end')
    
    r_errors = np.array(r_errors) 
    n_errors = np.array(n_errors)
    t_errors = np.array(t_errors)
    m_errors = np.array(m_errors)
    
    alm_r_errors = np.array(alm_r_errors) 
    alm_n_errors = np.array(alm_n_errors)
    alm_t_errors = np.array(alm_t_errors)
    alm_m_errors = np.array(alm_m_errors)
    
    hub_r_errors = np.array(hub_r_errors) 
    hub_n_errors = np.array(hub_n_errors)
    hub_t_errors = np.array(hub_t_errors)
    hub_m_errors = np.array(hub_m_errors)
    
    mean_r_errors = np.array(mean_r_errors) 
    mean_n_errors = np.array(mean_n_errors)
    mean_t_errors = np.array(mean_t_errors)
    mean_m_errors = np.array(mean_m_errors)
    
    deep_wind_info = np.array(deep_wind_info)
    alm_wind_info = np.array(alm_wind_info)
    hub_wind_info = np.array(hub_wind_info)
    mean_wind_info = np.array(mean_wind_info)
    
    file_name = os.path.join(save_dir, 'DeepALMtest_results.h5')
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('r_errors', data=r_errors)
        f.create_dataset('n_errors', data=n_errors)
        f.create_dataset('t_errors', data=t_errors)
        f.create_dataset('m_errors', data=m_errors)
        f.create_dataset('alm_r_errors', data=alm_r_errors)
        f.create_dataset('alm_n_errors', data=alm_n_errors)
        f.create_dataset('alm_t_errors', data=alm_t_errors)
        f.create_dataset('alm_m_errors', data=alm_m_errors)
        f.create_dataset('hub_r_errors', data=hub_r_errors)
        f.create_dataset('hub_n_errors', data=hub_n_errors)
        f.create_dataset('hub_t_errors', data=hub_t_errors)
        f.create_dataset('hub_m_errors', data=hub_m_errors)
        f.create_dataset('mean_r_errors', data=mean_r_errors)
        f.create_dataset('mean_n_errors', data=mean_n_errors)
        f.create_dataset('mean_t_errors', data=mean_t_errors)
        f.create_dataset('mean_m_errors', data=mean_m_errors)
        f.create_dataset('deep_wind_info', data=deep_wind_info)
        f.create_dataset('alm_wind_info', data=alm_wind_info)
        f.create_dataset('hub_wind_info', data=hub_wind_info)
        f.create_dataset('mean_wind_info', data=mean_wind_info)
    print('saved the result file')
    # Return summary statistics
    return r_errors, n_errors, t_errors, m_errors


def test(model, test_set, save, batch_size=600, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=32)
    
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    reference_error, normal_force_error, tangent_force_error, moment_torque_error = test_epoch(
        model=model_wrapper,
        loader=test_loader,
        save_dir=save)


def main(test_data_dir, save_dir, depth=100, growth_rate=12, efficient=True, batch_size=600, seed=None):
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
    
    test_set = WindDataset(test_data_dir)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)
    
    # load the latest model
    if not os.path.exists(save_dir):
        raise Exception('%s is not a dir' % save_dir)
    # model_files = os.listdir(save_dir)
    # model_files = [f for f in model_files if f.endswith('.pth')]
    # latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
    # latest_model_file = os.path.join(save_dir, latest_model_file)
    # model_state_dict  = torch.load(latest_model_file)
    model_state_dict = torch.load(os.path.join(save_dir, 'model_epoch0.pth'))
    model.load_state_dict(model_state_dict)

    # Train the model
    test(model=model, test_set=test_set, save=save_dir, batch_size=batch_size, seed=seed)
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
if __name__ == '__main__':
    test_data_dir  = './DeepALM_test_dataset'
    save_dir = './DeepALM_trained_model/adam_mse'
    # save_dir = './DeepALM_trained_model/adam_huber'
    # save_dir = './DeepALM_trained_model/sgd_mse'
    main(test_data_dir, save_dir)