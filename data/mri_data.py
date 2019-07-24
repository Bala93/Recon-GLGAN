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
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                undersampled_vol = cur_file['volus_{}x'.format(acceleration)]
                coords = cur_file['center_coord']
                num_slices = undersampled_vol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            except Exception as e:
                print(fname)
                print(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                input = np.pad(data['volus_{}x'.format(acceleration)][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                target = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                coords = data['center_coord'][slice]
                
                return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))
        except Exception as e:
            print(fname)
            print(e)
            
class SliceDataDev(Dataset):
	
    def __init__(self, root, acceleration):
        files = list(pathlib.Path(root).iterdir())
        #print(len(files))
        self.examples = []
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                undersampled_vol = cur_file['volus_{}x'.format(acceleration)]
                num_slices = undersampled_vol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            except Exception as e:
                print(fname)
                print(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                input = np.pad(data['volus_{}x'.format(acceleration)][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                target = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                coords = data['center_coord'].value

                return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords), fname.name, slice)
        except Exception as e:
            print(fname)
            print(e)