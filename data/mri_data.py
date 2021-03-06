
import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature

class SliceData(Dataset):


    def __init__(self, root, acceleration):

        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acceleration = acceleration

        for fname in sorted(files):

            cur_file = h5py.File(fname, 'r')
            undersampled_vol = cur_file['volus_{}x'.format(acceleration)]
            coords = cur_file['center_coord']
            num_slices = undersampled_vol.shape[2]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname, slice = self.examples[i]

        with h5py.File(fname, 'r') as data:

            # The (5,5) padding is done to accomodate image dimension to unet

            input = np.pad(data['volus_{}x'.format(self.acceleration)][:,:,slice], (5,5), 'constant', constant_values=(0,0))
            target = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
            coords = data['center_coord'][slice]
                
            return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))
            
class SliceDataDev(Dataset):
	
    def __init__(self, root, acceleration):

        files = list(pathlib.Path(root).iterdir())
        self.acceleration = acceleration
        self.examples = []
        for fname in sorted(files):

            cur_file = h5py.File(fname, 'r')
            undersampled_vol = cur_file['volus_{}x'.format(acceleration)]
            num_slices = undersampled_vol.shape[2]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):

        fname, slice = self.examples[i]

        with h5py.File(fname, 'r') as data:

            input = np.pad(data['volus_{}x'.format(self.acceleration)][:,:,slice], (5,5), 'constant', constant_values=(0,0))
            target = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
            coords = data['center_coord'].value

            return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords), fname.name, slice)
