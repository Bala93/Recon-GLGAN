"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        files = list(pathlib.Path(root).iterdir())
        #print(len(files))
        self.examples = []
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                fs_vol = cur_file['volfs']
                #fullysampled_vol = cur_file['volfs']
                coords = cur_file['center_coord']
                num_slices = fs_vol.shape[2]
                #print('erer')
                self.examples += [(fname, slice) for slice in range(num_slices)]
                #print('erer')
            except Exception as e:
                print('here123123')
                print(fname)
                print(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                #print(data.keys())
                input = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                #input /= np.max(input)
                target = np.pad(data['mask'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                #target /= np.max(target)
                #return (torch.from_numpy(input), torch.from_numpy(target), fname.name, slice)
                coords = data['center_coord'][slice]

                #edgemap = feature.canny(data['volfs'][:,:,slice])
                #edgemap = edgemap.astype(np.float)
                #print(np.unique(edgemap))
                #edgemap = np.pad(edgemap, (5,5), 'constant', constant_values=(0,0))

                '''
                cx, cy = coord[i]
                aux_input = np.zeros(input.shape)
                aux_input[cy-25:cy+25,cx-25:cx+25,:] = 1

                inp = np.dstack([input, aux_input])

                return (torch.from_numpy(inp), torch.from_numpy(target), torch.from_numpy(coords))   
                '''
                return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))
        except Exception as e:
            print('here123')
            print(fname)
            print(e)
            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        files = list(pathlib.Path(root).iterdir())
        #print(len(files))
        self.examples = []
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                fs_vol = cur_file['volfs']
                #fullysampled_vol = cur_file['volfs']
                #coords = cur_file['center_coord']
                num_slices = fs_vol.shape[2]
                #print('erer')
                self.examples += [(fname, slice) for slice in range(num_slices)]
                #print('erer')
            except Exception as e:
                print('here123123')
                print(fname)
                print(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                #print(data.keys())
                input = np.pad(data['volfs'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                #input /= np.max(input)
                target = np.pad(data['mask'][:,:,slice], (5,5), 'constant', constant_values=(0,0))
                #coords = data['center_coord'][slice]
                #target /= np.max(target)
                #edgemap = feature.canny(data['volfs'][:,:,slice])
                #edgemap = edgemap.astype(np.float)
                coords = data['center_coord'].value

                #print('fname slice ', fname, slice)
                #print('center_coord', coords)
                #edgemap = np.pad(edgemap, (5,5), 'constant', constant_values=(0,0))
                return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords), fname.name, slice)
                #coords = data['center_coord'][slice]
                #return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))
        except Exception as e:
            print('here123')
            print(fname)
            print(e)
            



'''

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        #print(len(files))
        for fname in sorted(files):
            try:
                kspace = h5py.File(fname, 'r')['kspace']
                num_slices = kspace.shape[0]
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
                kspace = data['kspace'][slice]
                target = data[self.recons_key][slice] if self.recons_key in data else None
                return self.transform(kspace, target, data.attrs, fname.name, slice)
        except Exception as e:
            print(fname)
            print(e)
    '''


'''
class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        #print(len(files))
        for fname in sorted(files):
            try:
                kspace = h5py.File(fname, 'r')['kspace']
                num_slices = kspace.shape[0]
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
                kspace = data['kspace'][slice]
                target = data[self.recons_key][slice] if self.recons_key in data else None
                return self.transform(kspace, target, data.attrs, fname.name, slice)
        except Exception as e:
            print(fname)
            print(e)
'''
