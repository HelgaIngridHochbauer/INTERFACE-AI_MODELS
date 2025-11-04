# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 17:24:46 2025

@author: Marc Frincu
"""

import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model
import argparse
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import pathlib
import glob
import cv2
from PIL import Image
import albumentations as A

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during testing.
    """
    def __init__(self, images):
        """
        Args:
            images: list of images for testing.
        """
        super(Dataset, self).__init__()

        self.images = images
        self.n_images = len(images)

        #self.mn = np.zeros((self.n_images))
        #self.std = np.zeros((self.n_images))
        #for i in range(self.n_images):
        #    self.mn[i] = np.mean(self.images[i], axis=(0,1))
        #    self.std[i] = np.std(self.images[i], axis=(0,1))

        #self.transform = A.Compose([A.RandomToneCurve(scale=0.2, p=0.5), A.RandomBrightnessContrast(p=0.2),])
                  
    def __getitem__(self, index):

        t = 'none'
        
        image = self.images[index]

        image_out = image

        #image_out = self.transform(image=image_out)['image']

        #image_out = (image_out - self.mn[index]) / self.std[index]

        return np.expand_dims(image_out, 0).astype('float32')

    def __len__(self):
        return self.n_images

# a simple custom collate function
def my_collate(batch):
    image = [torch.tensor(item[0]) for item in batch]
    return [image]

class Testing(object):
    """
    Testing class used for classifying the orientation of new images.
    The images should be resized to match the size of the training images.
    """
    def __init__(self, image_folder, model_file, gpu=0):
        """
        Args:
            image_folder (string): folder containing scaled testing images. 
                Image names should be ordered like output_INDEX.png
            model_file: the model used for classifying.
            gpu: index of the GPU to be used. If not found the CPU will be used instead.
        """
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda" if self.cuda else "cpu") #:cuda:{self.gpu}

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
           
        
        if (model_file is None):
            raise Exception("Model not provided for angle classification.") 
        else:
            self.checkpoint = '{0}'.format(model_file)
        
        self.model = model.Network(in_planes=1, n_kernel=16, n_out=360).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print("=> loading checkpoint '{}'".format(self.checkpoint))

        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

        len_data = 0            
        for f in tqdm(os.listdir(image_folder)):
            if os.path.isfile(os.path.join(image_folder, f)) and any(f.lower().endswith(ext) for ext in valid_extensions):        
                len_data += 1

        images = [None] * len_data

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        print("Reading images...")
        i=0
        for f in tqdm(os.listdir(image_folder)):
            if os.path.isfile(os.path.join(image_folder, f)) and any(f.lower().endswith(ext) for ext in valid_extensions):        
                images[i] = np.array(Image.open(os.path.join(image_folder, f)).convert("L"))
                i += 1
                # tmp = cv2.cvtColor(images[i], cv2.COLOR_RGB2YCrCb)
                # tmp[:, :, 0] = clahe.apply(tmp[:, :, 0])
                # images[i] = cv2.cvtColor(tmp, cv2.COLOR_YCrCb2RGB)
                # images[i] = images[i] / 255.0

        self.testing_size = len(images)
        
        self.testing_dataset = Dataset(images)
        
        self.testing_loader = torch.utils.data.DataLoader(self.testing_dataset, 
                                                          shuffle=False, 
                                                          collate_fn=my_collate)
        print("Images loaded...")

           
    def test(self, save_output=True):
        print("Running orientation classification...")
        t = tqdm(self.testing_loader) 
        predicted = torch.zeros(self.testing_size)

        loop = 0
        
        for batch_idx, images in enumerate(t):
            for i in range(len(images)):
                image = torch.from_numpy(np.asarray(images[i], dtype=np.float32)).to(self.device)

                out = self.model(image.unsqueeze(0)) # unsqueeze adds the batch dim to obtain a 4D input BxCxHxW = (Batch x Channel x Height x Width)

                predicted[loop] = out.max(1)[1] - 180
                                      
                loop+=1

        if save_output:
            np.savetxt('classification_output-predicted.txt',predicted, fmt='%.2f')

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Test neural network')
    parser.add_argument('--image_folder', help='Image folder')
    parser.add_argument('--model_file',  help='Neural net trained model')  
    parser.add_argument('--gpu', '--gpu', default=1, type=int,
                    metavar='GPU', help='GPU')

    
    parsed = vars(parser.parse_args())

    deepnet = Testing(image_folder=parsed['image_folder'], model_file=parsed['model_file'], gpu=parsed['gpu'])

    deepnet.test()

