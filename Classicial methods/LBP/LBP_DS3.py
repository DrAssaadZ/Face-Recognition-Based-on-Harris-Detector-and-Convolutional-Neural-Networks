'''
LBP facial recognition of the third dataset (AR Face dataset)
'''

import cv2 as cv
import numpy as np
from os import path
import os
from scipy.spatial import distance
from collections import Counter
import shutil
from skimage.feature import local_binary_pattern
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


# takes an image and returns its descriptor
def calculate_lbp_descriptor(gray_imag, raduis, n_points):

    lbp = local_binary_pattern(gray_imag, n_points, raduis, 'default')
    imgLBP = np.uint8(lbp)

    descriptor = cv.calcHist([imgLBP], [0], None, [256], [0, 256])  # calculating histogram

    return descriptor


def lbp_recognition(train_dataset_path, test_dataset_path, knn, tatal_tr_imgs, total_tst_imgs, train_img_nbr_class, tst_img_nbr_class, raduis, n_points):
    '''
    fonction that takes training and testing paths and prints the accuracy of the model

    params:
    train_dataset_path : train dataset path(string)
    test_dataset_path : test dataset path(string)
    knn : number of KNN used for score (int)
    tatal_tr_imgs : total number of train images in the dataset (int)
    total_tst_imgs : total number of train images in the dataset (int)
    train_img_nbr_class : the number of train images in one class (int)
    tst_img_nbr_class : the number of test images in one class (int)
    '''

    # for train -------------------------------------------------------------------------------
    train_dataset_files = os.listdir(train_dataset_path)

    # list that contains the images histogram descriptors
    train_images_hists = []
    # list that contains the image classes
    train_classes = []

    # for each train class folder
    for folder in train_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(train_dataset_path + folder)

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = np.array(
                cv.imread(train_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE))  # read as grayscale

            # calculating the image lbp descriptor
            img_lbp_hist = calculate_lbp_descriptor(img, raduis, n_points)

            # storing the img descriptors
            train_images_hists.append(img_lbp_hist)
            train_classes.append(folder)
        print("training class: ", folder)
    # print('training finished******************************')
    # for test ----------------------------------------------------------------------------------------------------------------------------------------
    test_dataset_files = os.listdir(test_dataset_path)

    # variabe that stores the distances between the test image and all the train images (should be of length of the training images)
    test_images_distances = np.zeros(tatal_tr_imgs)

    # vector that contains the prediction of classes (should be of test img nbr length)
    classes_vect = np.zeros(total_tst_imgs, dtype=int)

    # current real class
    current_class = 0

    # for each test class folder
    for folder in test_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(test_dataset_path + folder)

        # current image inside a folder
        current_image_index = 0

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = cv.imread(test_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE)  # read as grayscale

            # calculating the image lbp descriptor
            img_lbp_hist = calculate_lbp_descriptor(img, raduis, n_points)

            # compraing each test images with all the train images
            for i in range(tatal_tr_imgs):
                # calculating the distance between the test image and the i train image and storing the result in test_images_distances
                query_distance = distance.euclidean(img_lbp_hist, train_images_hists[i])
                test_images_distances[i] = query_distance

            # getting the K indexes of the smallest distances in the list
            smallest_indexes = sorted(range(len(test_images_distances)),
                                      key=lambda sub: test_images_distances[sub])[:knn]
            
            # calculating the classes from the smallest_indexes list : diving by the number of train images in a class | for exp : divinding by 7 which mean that
            # each 7 images there will be a new class
            img_possible_classes = list(map(lambda x: x // train_img_nbr_class, smallest_indexes))
            # most commun class in img_possible_classes
            image_class = Counter(img_possible_classes).most_common(1)
            print('predicted class :', image_class[0][0])
            print('real class :', current_class)
        
            # if the predicted class is correct
            if int(image_class[0][0]) == current_class:
                # the index of the classes vect : the current class * the total number of the test images in a class + the current image index
                classes_vect[current_class * tst_img_nbr_class + current_image_index] = 1

            current_image_index += 1

        current_class += 1

    # counting the most commun element in the classes_vect, could be 0 or 1
    accuracy_new = Counter(classes_vect).most_common(1)
    if accuracy_new[0][0] == 0:
        print(f'accuracy ({raduis}, {knn})', 1 - (accuracy_new[0][1] / total_tst_imgs))
    else:
        print(f'accuracy ({raduis}, {knn})', accuracy_new[0][1] / total_tst_imgs)

   

# main ----------------------------------------------------------

# train and test paths
train_path = '../../datasets/dataset3/orig_images/training/'
test_path = '../../datasets/dataset3/orig_images/testing/'


lbp_recognition(train_dataset_path=train_path, test_dataset_path=test_path, knn=1, tatal_tr_imgs=1400, 
            total_tst_imgs=1200, train_img_nbr_class=14, tst_img_nbr_class=12, raduis=1, n_points=8)