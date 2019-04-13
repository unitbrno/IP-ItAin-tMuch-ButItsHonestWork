#################################################################################
# Description:  File for training of clasification SVM
#               
# Authors:      Petr Buchal         <petr.buchal@lachub.cz>
#               Martin Ivanco       <ivancom.fr@gmail.com>
#               Vladimir Jerabek    <jerab.vl@gmail.com>
#
# Date:     2019/04/13
# 
# Note:     This source code is part of project created on UnIT HECKATHON
#################################################################################

from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from extracting_inception import create_graph, extract_features
from train_svm import train_svm_classifer

from tools import parse_data

if __name__ == '__main__':
    batch = 1000
    filenames = ["./images_png/"+f for f in listdir("./images_png") if isfile(join("./images_png", f))]

    #inception network model
    create_graph("./models/tensorflow_inception_graph.pb")

    labels = []
    for i, item in enumerate(filenames[:batch]):
        if item[-5] == 'T':
            labels.append(1)
        else:
            labels.append(0)

    #feature extraction
    features = extract_features(filenames[:batch], verbose=True)
    #SVM training
    train_svm_classifer(features, labels, "model.pkl")
