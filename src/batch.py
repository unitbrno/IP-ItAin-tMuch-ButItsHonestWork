#################################################################################
# Description:  Class for generating batch data for training 
#               clasification neural network
#               
# Authors:      Petr Buchal         <petr.buchal@lachub.cz>
#               Martin Ivanco       <ivancom.fr@gmail.com>
#               Vladimir Jerabek    <jerab.vl@gmail.com>
#
# Date:     2019/04/13
# 
# Note:     This source code is part of project created on UnIT HECKATHON
#################################################################################


import numpy as np
from image import Image
from tools import parse_data
import cv2
import random
from datetime import datetime

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

#Class storing data for augmentation
class DataSet(object):
    def __init__(self, ImageList):
        self.Images = ImageList
        self.p_images, self.labels, self. grand_truths = DataSet.strip_futilities(ImageList)
        random.seed(datetime.now())


    #Function for augmentation images
    #@param: imageArr   - is numpy array of images of shape (n, 1, height, width)
    #                                                        ^-- number of images
    #@param: grandArr   - is numpy array of grand truths images of shape (n, 1, height, width)
    #@param: batchSize  - is integer, which is less or equal 'n'
    #@return: new_img   - array of augmented images of shape (batchSize, 1, height, width)
    #@return: new_g_t   - array of augmented grand truths images of shape (batchSize, 1, height, width)
    @staticmethod
    def augmentImage(imageArr, grandArr, batchSize):
        
        for imgs in [imageArr, grandArr]:
            for img in imgs:
                h, w = img[0].shape
                img[0,0,:] = 0
                img[0,h-1,:] = 0
                img[0,:,0] = 0
                img[0,:,w-1] = 0

        shift = 0.2
        data_gen_args = dict(   data_format="channels_first",
                                rotation_range=90,
                                height_shift_range=shift,
                                width_shift_range=shift,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True)
        #creating Image generator same for image data and grand truths data
        img_dataGen = ImageDataGenerator(**data_gen_args)
        g_tr_dataGen = ImageDataGenerator(**data_gen_args)

        seed = random.randint(0,65535)
        for new_img in img_dataGen.flow(imageArr, seed=seed, batch_size=batchSize):
            new_img = np.uint16(new_img)
            break
        for new_g_t in g_tr_dataGen.flow(grandArr, seed=seed, batch_size=batchSize):
            new_g_t = np.uint16(new_g_t)
            break

        return new_img, new_g_t
        
    #Function for striping unneeded data from Image class
    #@param: data   - List of Image class       ^-- for more info see image.py
    #@return: images        - numpy array of images of shape (n, 1, height, width)      %note: 16-bit grayscale
    #@return: labels        - numpy array of booleans of shape (n,1), 
    #                           - 0 mean, that i-th image is doesn't contain ellipse
    #                           - 1 mean, that i-th image contains ellipse
    #@return: grand_truths  - numpy array of images of shape (n, 1, height, width)      %note: 16-bit grayscale
    @staticmethod
    def strip_futilities(data):
        images = []
        labels = []
        grand_truths = []
        for i, item in enumerate(data):
            images.append(np.array([item.processed_image]))
            labels.append(np.array([item.ellipse]))
            grand_truths.append(np.array([item.processed_ground_truths]))

        return np.array(images), np.array(labels), np.array(grand_truths)

    #Method for getting augmented data for training neural netowrk
    #@param batchSize   - nuber of images in one training epoche
    def getBatch(self, batchSize, isClassNet=True):
        img, g_truths = DataSet.augmentImage(self.p_images, self.grand_truths, batchSize)
        labels = []
        
        if batchSize > len(self.p_images):
            while len(img) != batchSize:
                curSize = batchSize - len(img)
                img2, g_truths2 = DataSet.augmentImage(self.p_images, self.grand_truths, curSize)
                img = np.concatenate((img, img2), axis=0)
                g_truths = np.concatenate((g_truths, g_truths2), axis=0)

        #get labels
        for i in range(len(img)):
            unique, counts = np.unique(g_truths[i], return_counts=True)
            if unique[-1] > 60000:
                labels.append(np.array([1]))
            else:
                labels.append(np.array([0]))

        if isClassNet:
            return img, np.array(labels)
        else:
            return img, g_truths


if __name__ == "__main__":
    trn_data = parse_data("./data/ground_truths_develop.csv", "./data/images/", "./data/ground_truths/")
    myData = DataSet(trn_data)
    counter = 0
    for e in range(100):
        x, y = myData.getBatch(32)
        for i, item in enumerate(x):
            print(np.shape(x))
            new = cv2.cvtColor(item[0], cv2.COLOR_GRAY2BGR)
            if y[i] == 0:
                cv2.imwrite("{}_F.png" .format(counter), new)
            if y[i] == 1:
                cv2.imwrite("{}_T.png" .format(counter), new)
            counter += 1
