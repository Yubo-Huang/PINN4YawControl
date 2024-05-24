import os
import h5py, time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WindDataset(Dataset):
    def __init__(self, dataset_dir, n_steps_past=20, n_steps_ahead=20):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(dataset_dir)
        self.n_steps_past = n_steps_past
        self.n_steps_ahead = n_steps_ahead
        self.seq_length = self.n_steps_past + self.n_steps_ahead
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
        wind_input, wind_label = self.load_sample(sample_path, sample_idx)
        return wind_input, wind_label

    def build_index_mapping(self):
        index_mapping = []
        global_idx = 0
        # index mapping remembers the number of samples with an increasing way
        # such as [100, 500, 510, 720, 1130]
        for file in self.file_list:
            sample_path = os.path.join(self.dataset_dir, file)
            time_length = self.get_timestep_in_a_file(sample_path) 
            if time_length >= self.seq_length:
                index_mapping.append(global_idx)
                num_samples = time_length - self.seq_length + 1
                global_idx += num_samples
            else:
                print('Warning: The file <{}> can not be opened'.format(file))
            
        index_mapping = np.array(index_mapping)
        total_samples = global_idx
        return index_mapping, total_samples

    def get_timestep_in_a_file(self, file):
        # Return the number of samples in a file
        # Modify this according to your file format
        with h5py.File(file, 'r') as data:
            sample = data['wind']
            num_samples = np.shape(sample)[0]
            
        return num_samples

    def load_sample(self, file_path, idx):
        # Load and preprocess the sample from the file
        # Modify this according to your file format
        # with np.load(file_path) as data:
        with h5py.File(file_path, 'r') as data:
            wind_input = data['wind'][idx:idx+self.n_steps_ahead, ...]
            wind_label = data['wind'][idx+self.n_steps_ahead:idx+self.n_steps_ahead+self.n_steps_past, ...]
        wind_input = torch.tensor(wind_input, dtype=torch.float32)
        wind_label = torch.tensor(wind_label, dtype=torch.float32)
        
        return wind_input, wind_label

if __name__ == '__main__':

    # Example usage:
    # root_dir = './WindForeTrainDataset/'
    root_dir = './WindFore_valid_dataset/'
    # root_dir = './HDF5dataset'
    dataset = WindDataset(root_dir)

    # Create a DataLoader to iterate over the dataset
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # Iterate over the dataloader
    ini = time.time()
    end = time.time()
    for batch_number, batch in enumerate(dataloader):
        if batch_number==0: 
            print('Time for prepartion is {}'. format(time.time()-end))
        else:
            print('Time for reading batch {} is {}'.format(batch_number, time.time()-end))
        end = time.time()
        if batch_number > 1000:
            print('time used for 1000 batch: {}'.format(time.time()-ini))
            break
        