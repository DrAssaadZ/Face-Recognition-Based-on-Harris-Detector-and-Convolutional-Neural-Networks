import cv2 as cv
import numpy as np
from os import path
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance
from collections import Counter
import shutil
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


# returns the mean face vector from images flattened vectors
def mean_face_vector(img_vects):
    # stacking the input image vector by colons
    stacked_vects = np.stack(img_vects, axis=1)
    mean_face_vect = []
    # for each colomn calculate the mean and store it on the mean face vect
    for i in range(len(img_vects[0])):
        mean_face_vect.append(np.mean(stacked_vects[i]) / len(img_vects))

    return mean_face_vect


def fisherface(train_dataset_path, test_dataset_path, nb_component, knn, total_tr_imgs, total_tst_imgs, train_img_nbr_class, tst_img_nbr_class):
    # for train -------------------------------------------------------------------------------
    train_dataset_files = os.listdir(train_dataset_path)

    # list that contains the images flattened vectors
    train_images_vects = []
    # list that contains the image classes
    train_classes = []

    # for each train class folder
    for folder in train_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(train_dataset_path + folder)

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = np.array(cv.imread(train_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE))
            # resized_img = cv.resize()
            # flattening the img [m * n * 1]
            flattened_img = img.flatten()
            # storing the img vect
            train_images_vects.append(flattened_img)
            train_classes.append(folder)

    # calculating the mean face vect
    train_mean_face_vect = mean_face_vector(train_images_vects)

    # substracting images from the mean face vect
    train_sub_image_vect = np.subtract(train_images_vects, train_mean_face_vect)

    # stacked matrix of subtracted image vectors
    train_st_sub_image_vect = np.stack(train_sub_image_vect)

    pca = PCA(n_components=nb_component)
    train_principalComponents = pca.fit_transform(train_st_sub_image_vect)
    # print(sum(pca.explained_variance_ratio_))
    # for testing ---------------------------------------------------------------------------------------
    test_dataset_files = os.listdir(test_dataset_path)

    
    test_images_distances = np.zeros(total_tr_imgs)

    # vector that contains the classes
    classes_vect = np.zeros(total_tst_imgs, dtype=int)

    current_class = 0

    # for each test class folder
    for folder in test_dataset_files:

        # listing files inside each folder ( class )
        folder_files = os.listdir(test_dataset_path + folder)

        current_image_index = 0

        # for each image in the folder class
        for file in folder_files:
            # read image
            img = np.array(cv.imread(test_dataset_path + folder + '/' + file, cv.IMREAD_GRAYSCALE)) # read as grayscale
            # resized_img = cv.resize()

            # flattening the img [m * n * 1]
            flattened_img = img.flatten()

            # substracting images from the mean face vect
            test_sub_image_vect = np.subtract(flattened_img, train_mean_face_vect)

            # reshaped test vector (1, ?)
            reshaped_test_vect = np.reshape(test_sub_image_vect, (1, len(test_sub_image_vect)))

            # calculating the test set principale components
            test_principalComponents = pca.transform(reshaped_test_vect)

            # compraing each test images with all the train images
            for i in range(total_tr_imgs):
                query_distance = distance.euclidean(test_principalComponents[0], train_principalComponents[i])
                test_images_distances[i] = query_distance

            # getting the indexes of the smallest distances in the list
            smallest_indexes = sorted(range(len(test_images_distances)), key=lambda sub: test_images_distances[sub])[:knn]

            # calculating the classes from the smallest_indexes list
            img_possible_classes = list(map(lambda x: x//train_img_nbr_class, smallest_indexes))

            # most commun class in img_possible_classes
            image_class = Counter(img_possible_classes).most_common(1)
            # print('image class :', image_class[0][0])

            # if the predicted class is correct
            if int(image_class[0][0]) == current_class:
                classes_vect[current_class * tst_img_nbr_class + current_image_index] = 1

            current_image_index += 1

        current_class += 1

    # calculating accuracy
    accuracy_new = Counter(classes_vect).most_common(1)
    if accuracy_new[0][0] == 0:
        print(f'accuracy ({nb_component}, {knn} ) =' , 1 - (accuracy_new[0][1] / total_tst_imgs))
    else:
        print(f'accuracy ({nb_component}, {knn} ) =' , accuracy_new[0][1] / total_tst_imgs)


# main ----------------------------------------------------------

# train and test paths
train_path = '../../datasets/dataset3/orig_images/training/'
test_path = '../../datasets/dataset3/orig_images/testing/'

fisherface(train_dataset_path=train_path, test_dataset_path=test_path, nb_component=70, knn=1, total_tr_imgs=1400,
                total_tst_imgs=1200, train_img_nbr_class=14, tst_img_nbr_class=12)
        


