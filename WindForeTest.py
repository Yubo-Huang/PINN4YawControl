import os
import matplotlib.pyplot as plt
import time
import random
import h5py
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

def visual_data(model, loader, n_step_ahead, save_dir):
    # Model on eval mode
    model.eval()

    with torch.no_grad():
        for batch_samples in loader:
            [input, label] = batch_samples
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()

            # compute output
            pred = model(input, n_step_ahead)
            break
    
    net_input  = input.cpu().numpy()
    net_output = pred.cpu().numpy()
    net_label  = label.cpu().numpy()
    
    file_name = os.path.join(save_dir, 'WindFore_results.h5')
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('net_input', data=net_input)
        f.create_dataset('net_output', data=net_output)
        f.create_dataset('net_label', data=net_label)
    
    result = {'input':net_input, 'output':net_output, 'label':label}       
    return result

def main(valid_data_dir, save_dir, seed=None, restore=True):

    # Models
    model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=3)
    print(model)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)
    
    valid_set = WindDataset(valid_data_dir)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=opt.batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=32)

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

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    visual_data(model=model_wrapper, loader=valid_loader, n_step_ahead=opt.n_step_ahead, save_dir=save_dir)
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
    valid_data_dir = './WindFore_valid_dataset'
    save_dir = './WindFore_trained_model'
    main(valid_data_dir, save_dir)

