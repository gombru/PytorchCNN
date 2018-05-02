from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import customTransform
from PIL import Image

class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, split, Rescale, RandomCrop, Mirror):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.Rescale = Rescale
        self.RandomCrop = RandomCrop
        self.Mirror = Mirror


        split_f  = '{}/{}.txt'.format(root_dir, split)
        self.indices = open(split_f, 'r').read().splitlines()

        # Save image labels (one hot encoding)
        self.labels = [int(i.split(',', 1)[1]) for i in self.indices]

        # self.labels_onehot = []
        # for l in labels:
        #     onehot = np.zeros(10)
        #     onehot[l] = 1
        #     self.labels_onehot.append(onehot)

        # Save image names
        self.indices = [i.split(',', 1)[0] for i in self.indices]
        print("Num images in " + split + " --> " + str(len(self.indices)))

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        img_name = self.root_dir + '/img_resized_1M/cities_instagram/' + self.indices[idx] + '.jpg'
        try:
            image = Image.open(img_name)
            # print("FOUND " + img_name)
        except:
            # print("Img file not found, using hardcoded " + img_name)
            img_name = self.root_dir + '/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
            image = Image.open(img_name)

        try:
            if self.Rescale != 0:
                image = customTransform.Rescale(image,self.Rescale)

            if self.RandomCrop != 0:
                image = customTransform.RandomCrop(image,self.RandomCrop)

            if self.Mirror:
                image = customTransform.Mirror(image)

            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        except:
            print("Error on data aumentation, using hardcoded")
            img_name = self.root_dir + '/img_resized_1M/cities_instagram/london/1481255189662056249.jpg'
            image = Image.open(img_name)
            if self.RandomCrop != 0:
                image = customTransform.RandomCrop(image,self.RandomCrop)
            im_np = np.array(image, dtype=np.float32)
            im_np = customTransform.PreprocessImage(im_np)

        out_img = np.copy(im_np)

        label = torch.from_numpy(np.array([int(self.labels[idx])]))
        label = label.type(torch.LongTensor)

        return torch.from_numpy(out_img), label