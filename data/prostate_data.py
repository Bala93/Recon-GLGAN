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
import glob
import os 

class SliceDataDyn(Dataset):
    def __init__(self, file_names,object_type, mode):
        self.file_names  = file_names
        self.object_type = object_type
        self.mode = mode
        #print (self.object_type)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
    
        img_file_name = self.file_names[idx]
        #print(img_file_name)
        with h5py.File(img_file_name, 'r') as data:
#             print(data.keys())
            image = data['img'].value
            # print(img_file_name,image.shape,type(image))
            image = np.transpose(image,[2,0,1])
            mask = data['mask'].value
            mask = np.transpose(mask,[2,0,1])
            # print(type(data['dist'].value),type(mask),type(image))
#             dist = data['dist'].value
            # coord_final = np.zeros([2,30]) #Assuming max no.of.coords of interest in an image is 30.
            # slice_no = np.array([0,0])

            if len(data.keys()) == 4:
                # [h,w] = data['coord'].shape
                # slice_no = np.array([h,w])
                # coord_final[:h,:w] = data['coord']
                # print(type(data['coord'].value))
                coord = data['coord'].value
                bbox = data['bbox'].value
                # print(coord)
#             print(bbox,coord.shape,image.shape,mask.shape)
#             seed = np.random.randint(0,8,1)[0]
#             print('seed=',seed)
#             plt.figure()
#             plt.subplot(121)
#             plt.imshow(image[seed])
#             plt.subplot(122)
#             plt.imshow(mask[seed])
#             plt.show()
            coord = torch.from_numpy(coord)
            # else:
            #     coord = np.zeros([1,1])

            # image = np.transpose(image,[1,2,0])
            image = np.expand_dims(image,axis=1)
            image = torch.from_numpy(image)
            # print(image.shape)
            # data_transforms = transforms.Compose([
            #      transforms.ToTensor()])
                #  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            # image = data_transforms(image)
            # print(image.shape)
            mask[mask == 255.] = 1
            mask = mask.astype(np.uint8)
            #print(coord_final,coord_final.shape)
            # print ()
        if self.mode == 'train':    
            # print (image.shape,mask.shape,coord.shape,dist.shape)
            # print (image.dtype,mask.dtype,coord.dtype,dist.dtype)
            return img_file_name,image.float(),torch.from_numpy(np.expand_dims(mask, 1)).long(), coord, bbox
        if self.mode == 'valid':
            return img_file_name,image.float(),torch.from_numpy(np.expand_dims(mask, 1)).long()
 

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR img slices.
    """

    def __init__(self, root):
        #files = list(pathlib.Path(root).iterdir())
        files = glob.glob(os.path.join(root,'*.h5'))
        #print(len(files))
        self.examples = sorted(files)
        '''
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                fs_vol = cur_file['img']
                coords = cur_file['coord']
                num_slices = fs_vol.shape[2]
                #print('erer')
                self.examples += [fname]
                #print('erer')
            except Exception as e:
                print('init_train')
                print(fname)
                print(e)
        '''
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                #print(data.keys())
                input = data['img'].value
                #input /= np.max(input)

                target = data['mask'].value.astype(np.uint8)
                target[target==255] = 1
                #target /= np.max(target)
                #return (torch.from_numpy(input), torch.from_numpy(target), fname.name, slice)
                coords = data['coord'].value

                #edgemap = feature.canny(data['img'][:,:,slice])
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
            print('Problem with the h5 files created : Please check')
            print(fname)
            print(e)
            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR img slices.
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
        self.examples = sorted(files)
        '''
        for fname in sorted(files):
            try:
                cur_file = h5py.File(fname, 'r')
                fs_vol = cur_file['img']
                self.examples += [fname]
                #print('erer')
            except Exception as e:
                print('init_dev')
                print(fname)
                print(e)
        '''
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]
        try:
            with h5py.File(fname, 'r') as data:
                #print(data.keys())
                input = data['img'].value
                #input /= np.max(input)
                target = data['mask'].value.astype(np.uint8)
                target[target==255] = 1
                #coords = data['coord'][slice]
                #target /= np.max(target)
                #edgemap = feature.canny(data['img'][:,:,slice])
                #edgemap = edgemap.astype(np.float)
                coords = data['coord'].value

                #print('fname slice ', fname, slice)
                #print('coord', coords)
                #edgemap = np.pad(edgemap, (5,5), 'constant', constant_values=(0,0))
                return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords), fname.name)
                #coords = data['coord'][slice]
                #return (torch.from_numpy(input), torch.from_numpy(target), torch.from_numpy(coords))
        except Exception as e:
            print('getitem Dev')
            print(fname)
            print(e)
            



'''

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR img slices.
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
    A PyTorch Dataset that provides access to MR img slices.
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
