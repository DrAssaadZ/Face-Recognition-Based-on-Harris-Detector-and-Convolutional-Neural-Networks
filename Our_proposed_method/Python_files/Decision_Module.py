from keras.preprocessing import image
import numpy as np
import os
from collections import Counter


class decisionModule:
    @staticmethod
    def Calculate_model_accuracy( classifier, dataset_path, nbr_classes, tst_img_per_class):

        # calculating the total number of test images
        total_tst_imgs = nbr_classes * tst_img_per_class

        # listing file directories
        dataset_files = os.listdir(dataset_path)

        # vector to store the class of each test image
        classes_vect = np.zeros(total_tst_imgs, dtype=int)

        current_class = 0

        # looping through the folder files
        for folder in dataset_files:
            folder_path = dataset_path + folder

            # looping through the classes folders
            sub_folder = os.listdir(folder_path)

            # index of the current image
            current_image_index = 0

            # looping through the images folders
            for file in sub_folder:

                # listing files in
                ROI_list = os.listdir(folder_path + '/' + file)

                ROI_vect = np.zeros(len(ROI_list))
                current_ROI_index = 0

                # looping through the region of interest images
                for ROI_img in ROI_list:
                    # path for the current region of interest
                    ROI_img_path = folder_path + '/' + file + '/' + ROI_img
                    # loading the ROI
                    img = image.load_img(ROI_img_path)
                    img = image.img_to_array(img) / 255.0
                    img = img.reshape((1,) + img.shape)
                    # predicting the ROI class
                    img_class = classifier.predict_classes(img)
                    classname = int(img_class[0])
                    # apprending the ROI img class to the ROI vector
                    ROI_vect[current_ROI_index] = classname
                    current_ROI_index += 1
                # the ROI class is the class that is repeated the most
                ROI_image_class = Counter(ROI_vect).most_common(1)

                # checking if the predicted class is the same as the real class for the ROI
                if int(ROI_image_class[0][0]) == current_class:
                    classes_vect[current_class * tst_img_per_class + current_image_index] = 1

                current_image_index += 1

            current_class += 1

        accuracy_new = Counter(classes_vect).most_common(1)
        if accuracy_new[0][0] == 0:
            print('The model final accuracy is : ', 1 - (accuracy_new[0][1] / total_tst_imgs)) 
        else:
            print('The model final accuracy is :', accuracy_new[0][1] / total_tst_imgs) 
