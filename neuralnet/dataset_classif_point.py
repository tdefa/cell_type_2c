import os

import cv2
import torch
from matplotlib import pyplot
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from skimage.transform import resize

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
uno  = UnNormalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])


class SpotDataset(Dataset):
    

    def __init__(self, root_dir, transform=None,  resize_size= None, normalize = True, add_channel = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir) #image directory
        self.transform = transform
        self.get_min_max_parameter()
        self.resize_size = resize_size
        self.normalize = normalize
        self.add_chanel = add_channel
        if self.normalize == 2:
             self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip()])

        if self.normalize == 3:
             self.transform0 = transforms.Compose([
                 transforms.RandomHorizontalFlip(),
                   transforms.Normalize([0.456, 0.406], [ 0.224, 0.225])
            ])

             self.transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                  transforms.Normalize([0.456, 0.406], [ 0.224, 0.225]),
                transforms.GaussianBlur(kernel_size =5, sigma=(0.01, 1))])
        if self.normalize == 4:
             self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                 transforms.Normalize([0.456, 0.406], [ 0.224, 0.225]),
                transforms.GaussianBlur(kernel_size =5, sigma=(0.01, 1))])


    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def get_min_max_parameter(self):
        min_d = 4095
        max_d = 0
        min_d0 = 4095
        max_d0= 0
        min_d1 = 4095
        max_d1 = 0
        len_dataset = len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])
        for i in range(len_dataset):
            idx = str(i)
            idx = idx.zfill(7)
            full_name = next(self.root_dir.glob(f'*{idx}*'))
            input_image = np.load(str(full_name))
            if input_image.max() > max_d:
                max_d = input_image.max()
            if input_image.min() < min_d:
                min_d = input_image.min()
            if input_image.max() > max_d:
                max_d = input_image.max()
            if input_image.min() < min_d:
                min_d = input_image.min()
            if input_image.max() > max_d:
                max_d = input_image.max()
            if input_image.min() < min_d:
                min_d = input_image.min()
        self.min = min_d
        self.max = max_d
        self.min0 = min_d0
        self.max0 = max_d0
        self.min1 = min_d1
        self.max1 = max_d1


    def __getitem__(self, idx):
        idx = str(idx)
        idx = idx.zfill(7)
        full_name = next(self.root_dir.glob(f'*{idx}*'))
        input_image = np.load(str(full_name))
        if "fake" in str(full_name):
            label = 1
        else:
            label =  0 

        if self.resize_size is not None and self.resize_size != 0:
            input_image = resize(input_image , (2, self.resize_size , self.resize_size), preserve_range = True)
        if self.normalize == 1:
            input_image = (input_image -self.min) / (self.max - self.min) #min max normalization
            input_image = torch.tensor(input_image)
            input_image = self.transform(input_image)
        elif self.normalize == 2:
            input_image = input_image.astype('float64')
            input_image[0] = (input_image[0] - input_image[0].min()) / (input_image[0].max() - input_image[0].min()) #min max normalization
            input_image[1] = (input_image[1] - input_image[1].min()) / (input_image[1].max() - input_image[1].min()) #min max normalization
            input_image[0] = (input_image[0] - input_image[0].mean()) / input_image[0].std() #
            input_image[1] = (input_image[1] - input_image[1].mean()) / input_image[1].std() # min max normalization


            input_image = torch.tensor(input_image)
            input_image = self.transform(input_image)

        elif self.normalize == 3:
            if label == 0:
                input_image = (input_image -self.min) / (self.max - self.min) #min max normalization
                input_image = torch.tensor(input_image)
                input_image = self.transform0(input_image)
            if label == 1:
                input_image = (input_image -self.min) / (self.max - self.min) #min max normalization
                input_image = torch.tensor(input_image)
                input_image = self.transform1(input_image)
        elif self.normalize == 4:
                input_image = (input_image -self.min) / (self.max - self.min) #min max normalization
                input_image = torch.tensor(input_image)
                input_image = self.transform(input_image)



        else:
            input_image = torch.tensor(input_image.astype('float64'))

        if self.add_chanel:
           input_image = torch.cat((input_image,torch.zeros((1,input_image.shape[1], input_image.shape[2]))))




        return input_image, {'label' :  label}