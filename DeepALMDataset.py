import os
import torch
import time
import h5py
import threading
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WindDataset(Dataset):
    def __init__(self, dataset_dir, save_step=2000):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(dataset_dir)
        # num_files = len(self.file_list)
        # self.index_mapping = np.arange(0, save_step * num_files, save_step)
        # self.total_samples = save_step * num_files
        self.index_mapping, self.total_samples = self.build_index_mapping()

    def __len__(self):
        return self.total_samples

    
    def __getitem__(self, idx):
        
        # find the file that saves sample idx
        file_idx = np.sum(self.index_mapping < (idx+1)) - 1 
        # determine the sample number in the target file
        sample_idx = idx - self.index_mapping[file_idx]
        # extract the file name
        file = self.file_list[file_idx]
        # generate the file fold 
        sample_path = os.path.join(self.dataset_dir, file)
        # load sample idx or sample_idx from the target file
        body_velocity, normal_vector, tangent_vector, normal_coef, tangent_coef, moment_coef, normal_force, tangent_force, moment_torque, uniform_wind, wind_blade_info = self.load_sample(sample_path, sample_idx)
        return body_velocity, normal_vector, tangent_vector, normal_coef, tangent_coef, moment_coef, normal_force, tangent_force, moment_torque, uniform_wind, wind_blade_info

    def build_index_mapping(self):
        index_mapping = []
        global_idx = 0
        # index mapping remembers the number of samples with an increasing way
        # such as [100, 500, 510, 720, 1130]
        for file in self.file_list:
            index_mapping.append(global_idx)
            sample_path = os.path.join(self.dataset_dir, file)
            num_samples = self.get_sample_number_in_a_file(sample_path)
            global_idx += num_samples
        index_mapping = np.array(index_mapping)
        total_samples = global_idx
        return index_mapping, total_samples

    def get_sample_number_in_a_file(self, file):
        # Return the number of samples in a file
        # Modify this according to your file format
        # with np.load(file) as data:
        with h5py.File(file, 'r') as data:
            sample = data['normal_force']
            num_samples = np.shape(sample)[0]
            
        return num_samples

    def load_sample(self, file_path, idx):
        # Load and preprocess the sample from the file
        # Modify this according to your file format
        # with np.load(file_path) as data:
        with h5py.File(file_path, 'r') as data:
            body_velocity = data['body_velocity'][idx, ...]
            normal_vector = data['normal_vector'][idx, ...]
            tangent_vector = data['tangent_vector'][idx, ...]
            normal_coef = data['normal_coef'][idx, ...]
            tangent_coef = data['tangent_coef'][idx, ...]
            moment_coef = data['moment_coef'][idx, ...]
            normal_force = data['normal_force'][idx, ...]
            tangent_force = data['tangent_force'][idx, ...]
            moment_torque = data['moment_torque'][idx, ...]
            uniform_wind = data['uniform_wind'][idx, ...]
            wind_blade_info = data['wind_blade_info'][idx, ...]
            
        body_velocity = torch.tensor(body_velocity, dtype=torch.float32)
        normal_vector = torch.tensor(normal_vector, dtype=torch.float32)
        tangent_vector = torch.tensor(tangent_vector, dtype=torch.float32)
        normal_coef = torch.tensor(normal_coef, dtype=torch.float32)
        tangent_coef = torch.tensor(tangent_coef, dtype=torch.float32)
        moment_coef = torch.tensor(moment_coef, dtype=torch.float32)
        normal_force = torch.tensor(normal_force, dtype=torch.float32)
        tangent_force = torch.tensor(tangent_force, dtype=torch.float32)
        moment_torque = torch.tensor(moment_torque, dtype=torch.float32)
        uniform_wind = torch.tensor(uniform_wind, dtype=torch.float32)
        wind_blade_info = torch.tensor(wind_blade_info, dtype=torch.float32)
        
        return body_velocity, normal_vector, tangent_vector, normal_coef, tangent_coef, moment_coef, normal_force, tangent_force, moment_torque, uniform_wind, wind_blade_info


def read_file(file_path, data_array, batch_index, sample_id):
    with h5py.File(file_path, 'r') as data:
        keys = data.keys()
        for key in keys:
            data_array[key][batch_index, ...] = data[key][sample_id, ...]

class WindDataLoader():
    def __init__(self, dataset_dir, batch_size=64, shuffle=False):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(dataset_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_mapping, self.total_samples = self.build_index_mapping()
        self.num_batch = self.total_samples // batch_size
        # generate_batch_sample_id will set self.current_batch and self.batch_sample_id
        self.generate_batch_sample_id()
        self.batch_sample = {}
        self.set_batch_sample_shape()
        print('initialisation completed')
    
    def __len__(self):
        return self.total_samples
    
    def build_index_mapping(self):
        index_mapping = []
        global_idx = 0
        # index mapping remembers the number of samples with an increasing way
        # such as [100, 500, 510, 720, 1130]
        for file in self.file_list:
            index_mapping.append(global_idx)
            sample_path = os.path.join(self.dataset_dir, file)
            num_samples = self.get_sample_number_in_a_file(sample_path)
            global_idx += num_samples
        index_mapping = np.array(index_mapping)
        total_samples = global_idx
        return index_mapping, total_samples
    
    def set_batch_sample_shape(self):
        file = os.path.join(self.dataset_dir, self.file_list[0])
        with h5py.File(file, 'r') as data:
            keys = data.keys()
            for key in keys:
                shape = list(np.shape(data[key]))
                shape[0] = self.batch_size
                self.batch_sample[key] = np.zeros(shape)
    
    def generate_batch_sample_id(self):
        num_batch = self.num_batch
        batch_size = self.batch_size
        shuffle = self.shuffle
        batch_sample_id = np.arange(num_batch*batch_size)
        if shuffle:
            np.random.shuffle(batch_sample_id)
        batch_sample_id = batch_sample_id.reshape(num_batch, batch_size)
            
        self.current_batch = 0
        self.batch_sample_id = batch_sample_id
    
    def get_batch_sample(self):
        threads = []
        current_batch = self.current_batch
        batch_sample = self.batch_sample
        batch_id = self.batch_sample_id[current_batch]
        for i, batch_idx in enumerate(batch_id):
            file_idx = np.sum(self.index_mapping < (batch_idx+1)) - 1 
            # determine the sample number in the target file
            sample_idx = batch_idx - self.index_mapping[file_idx]
            # extract the file name
            file = self.file_list[file_idx]
            # generate the file fold 
            sample_path = os.path.join(self.dataset_dir, file)
            thread = threading.Thread(target=read_file, args=(sample_path, batch_sample, i, sample_idx))
            threads.append(thread)
            thread.start()
        
        # wait until all threads complete their tasks
        for thread in threads:
            thread.join()
        
        # delete all threads to release resource
        for thread in threads:
            del thread
            
        self.current_batch = self.current_batch + 1
        if self.current_batch == self.num_batch:
            self.generate_batch_sample_id()
        
        data = batch_sample
        body_velocity = data['body_velocity']
        normal_vector = data['normal_vector']
        tangent_vector = data['tangent_vector']
        normal_coef = data['normal_coef']
        tangent_coef = data['tangent_coef']
        moment_coef = data['moment_coef']
        normal_force = data['normal_force']
        tangent_force = data['tangent_force']
        moment_torque = data['moment_torque']
        uniform_wind = data['uniform_wind']
        wind_blade_info = data['wind_blade_info']
        
        body_velocity = torch.tensor(body_velocity, dtype=torch.float32)
        normal_vector = torch.tensor(normal_vector, dtype=torch.float32)
        tangent_vector = torch.tensor(tangent_vector, dtype=torch.float32)
        normal_coef = torch.tensor(normal_coef, dtype=torch.float32)
        tangent_coef = torch.tensor(tangent_coef, dtype=torch.float32)
        moment_coef = torch.tensor(moment_coef, dtype=torch.float32)
        normal_force = torch.tensor(normal_force, dtype=torch.float32)
        tangent_force = torch.tensor(tangent_force, dtype=torch.float32)
        moment_torque = torch.tensor(moment_torque, dtype=torch.float32)
        uniform_wind = torch.tensor(uniform_wind, dtype=torch.float32)
        wind_blade_info = torch.tensor(wind_blade_info, dtype=torch.float32)
        
        return body_velocity, normal_vector, tangent_vector, normal_coef, tangent_coef, moment_coef, normal_force, tangent_force, moment_torque, uniform_wind, wind_blade_info

    def get_sample_number_in_a_file(self, file):
        # Return the number of samples in a file
        # Modify this according to your file format
        # with np.load(file) as data:
        with h5py.File(file, 'r') as data:
            sample = data['normal_force']
            num_samples = np.shape(sample)[0]
            
        return num_samples

if __name__ == '__main__':

    # Example usage:
    root_dir = 'DeepALM_train_dataset'
    # root_dir = 'HDF5dataset'
    save_step = 2000
    dataset = WindDataset(root_dir, save_step)

    # Create a DataLoader to iterate over the dataset
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # Iterate over the dataloader
    ini = time.time()
    end = time.time()
    for batch_number, batch in enumerate(dataloader):
        # Training code here
        # print(type(batch))
        # if batch_number==0: 
        #     print('Time for prepartion is {}'. format(time.time()-end))
        # else:
        #     print('Time for reading batch {} is {}'.format(batch_number, time.time()-end))
        # end = time.time()
        if batch_number > 10000:
            print('time used for 1000 batch: {}'.format(time.time()-ini))
            break
    
    # loader = WindDataLoader(root_dir, batch_size, shuffle=True)
    # start = time.time()
    # for i in range(10):
    #     batch = loader.get_batch_sample()
    # print('Time used for one batch is {}'.format(time.time()-start))
     