import cv2 as cv
import os
from os import path
import numpy as np
import cornerDetectors
import time


class Preprocessing():

	def get_region_of_interest(self, img):
		'''
		function that takes an img and returns the regions of interest of it (as a list)
		'''

	    # getting the image shape
		w, h = img.shape[1], img.shape[0]

	    # converting the image to grayscale
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	    # detecting the corners (interest points) in an image
		corners = cornerDetectors.shi_tomas_detector(gray, 28, 20) # to get x and y we need to use x,y = corner.ravel()
		# corners = cornerDetectors.sift_detector(gray, 28) # to get x and y we need to use corner.pt[:] in the next loop
	    # corners = cornerDetectors.surf_detector(gray) # to get x and y we need to use corner.pt[:] in the next loop
	    # corners = cornerDetectors.fast_detector(gray) # to get x and y we need to use corner.pt[:] in the next loop

	    # list that contains all the regions of interest of an input image
		image_block = []

		for corner in corners:
			x, y = corner.ravel()  # this line changes depending on the detector
			x = x - 16
			y = y - 16

	        # dealing with interest point at the corner of the image
			if x < 0:
				x = 0

			if x + 32 > w:
				x = w - 32

			if y < 0:
				y = 0

			if y + 32 > h:
				y = h - 32

	        # cropping image blocks 32x32 around all the interest points
			cropped_img = img[int(y):int(y + 32), int(x):int(x + 32)]

	        # storing all the regions of interset in a list 
			image_block.append(cropped_img)

		return image_block

	def prepare_dataset1(self, dataset_path):

		os.mkdir('datasets/dataset1')

		# creating output folders
		os.mkdir('datasets/dataset1/orig_images')
		os.mkdir('datasets/dataset1/orig_images/training')
		os.mkdir('datasets/dataset1/orig_images/testing')

		
		# creating region of interest dataset folders
		os.mkdir('datasets/dataset1/ROI_images')
		os.mkdir('datasets/dataset1/ROI_images/ROI_dataset')
		os.mkdir('datasets/dataset1/ROI_images/ROI_dataset/training')
		os.mkdir('datasets/dataset1/ROI_images/ROI_dataset/testing')

		# creating KNN dataset
		os.mkdir('datasets/dataset1/ROI_images/KNN_dataset')

		# variable that stores the current output folder (either training or testing), for the original images and the ROI images
		output_folder = 'datasets/dataset1/orig_images/training/'
		ROI_output_folder = 'datasets/dataset1/ROI_images/ROI_dataset/training/'
		KNN_output_folder = 'datasets/dataset1/ROI_images/KNN_dataset/'

	    # counter used to count images in each class
		counter = 0

	    # listing dataset files
		dataset_files = os.listdir(dataset_path)

	    # for each class folder in the original dataset path
		for folder in dataset_files:

	        # folder files [1.ppm, 10.ppm, 2.ppm ...]
			folder_files = os.listdir(dataset_path + folder)
	        # folder files sorted [1.ppm, 2.ppm, 3.ppm ...] 
			folder_files_sorted = sorted(folder_files, key=lambda x: int(os.path.splitext(x)[0]))

	        # for each image in the
			for image in folder_files_sorted:
	        	# create the class folder if it doesn't exist
				if not path.exists(output_folder + folder):
					os.mkdir(output_folder + folder)

	            # reading then saving the original image in its corresponding folder
				img = cv.imread(dataset_path + folder + '/' + image)
				cv.imwrite(output_folder + folder + '/' + folder + '_' + image[:-4] + '.jpg', img)

				# create the output class folder if it doesn't exist for the region of interests
				if not path.exists(ROI_output_folder + folder):
					os.mkdir(ROI_output_folder + folder)

				# getting its regions of interest of the images 
				ROI_imgs = self.get_region_of_interest(img)

            	# save each region of interest in its corresponding folder
				for i in range(len(ROI_imgs)):
					cv.imwrite(ROI_output_folder + folder + '/' + folder + '_' + image[:-4] + '_' + str(i) + '.jpg', ROI_imgs[i])

					# creating a folder for region of interest in the testing images (used for the KNN dataset)
					if 'testing' in ROI_output_folder:
						# creating the class folder with only the second part of the file_split : 8.jpg
						if not path.exists(KNN_output_folder +  folder + '/' + image[:-4] ):
							os.makedirs(KNN_output_folder +  folder + '/' + image[:-4])

						cv.imwrite(KNN_output_folder +  folder + '/' + image[:-4] + '/' + image[:-4] + '.' + str(i) + '.jpg', ROI_imgs[i])

				counter += 1

	            # first 7 images are used for training in each class
				if counter == 7:
					output_folder = 'datasets/dataset1/orig_images/testing/'
					ROI_output_folder = 'datasets/dataset1/ROI_images/ROI_dataset/testing/'

	            # last 3 images are used for testing in each class
				if counter == 10:
					output_folder = 'datasets/dataset1/orig_images/training/'
					ROI_output_folder = 'datasets/dataset1/ROI_images/ROI_dataset/training/'
					counter = 0

	def prepare_dataset2(self, dataset_path):
		'''
		function that takes the original dataset path and creates 5 folds, 
		each fold has the splitted dataset (train and test) along with the regions of interest images  
		'''
		# creating output folders
		os.mkdir('datasets/dataset2')

		# variable used to separate the folds starting from the second fold exp : fold 2 : 3, fold 3: 6 ...
		fold_counter = 3

		# creating the 5 folds folders
		for fold in range(1, 6):
			# creating the fold
			os.mkdir('datasets/dataset2/fold ' + str(fold))

			# creating output folders
			os.mkdir(f'datasets/dataset2/fold {fold}/orig_images')
			os.mkdir(f'datasets/dataset2/fold {fold}/orig_images/training')
			os.mkdir(f'datasets/dataset2/fold {fold}/orig_images/testing')

			os.mkdir(f'datasets/dataset2/fold {fold}/ROI_images')
			# creating region of interest dataset folders
			
			os.mkdir(f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset')
			os.mkdir(f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/training')
			os.mkdir(f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/testing')

			# creating KNN dataset
			os.mkdir(f'datasets/dataset2/fold {fold}/ROI_images/KNN_dataset')

			
			KNN_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/KNN_dataset/'
			# if this is the first fold then the output folder starts with 3 testing images, else it starts with training
			if fold == 1:
				output_folder = f'datasets/dataset2/fold {fold}/orig_images/testing/'
				ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/testing/'
			else: 
				output_folder = f'datasets/dataset2/fold {fold}/orig_images/training/'
				ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/training/'
				
			# counter used to count images in each class
			counter = 0
			
			# listing dataset files
			dataset_files = os.listdir(dataset_path)

			# for each images in the original dataset
			for image in dataset_files:
				# image name split[0] is used as a folder/class name
				img_name_split = image.split('_')

				# creating the class folder if it doesnt exist
				if not path.exists(output_folder + img_name_split[0]):
					os.mkdir(output_folder + img_name_split[0])	

				# reading then saving the original image in its corresponding folder
				img = cv.imread(dataset_path + image)
				cv.imwrite(output_folder + img_name_split[0] + '/' + image , img)

				# # create the output class folder if it doesn't exist for the region of interests
				if not path.exists(ROI_output_folder + img_name_split[0]):
					os.mkdir(ROI_output_folder + img_name_split[0])

				# getting its regions of interest of the images 
				ROI_imgs = self.get_region_of_interest(img)

				# save each region of interest in its corresponding folder
				for i in range(len(ROI_imgs)):
					cv.imwrite(ROI_output_folder + img_name_split[0] + '/' + image[:-4] + '_' + str(i) + '.jpg', ROI_imgs[i])

					# creating a folder for region of interest in the testing images (used for the KNN dataset)
					if 'testing' in ROI_output_folder:
						# creating the class folder with only the second part of the file_split : 8.jpg
						if not path.exists(KNN_output_folder +  img_name_split[0] + '/' + image[:-4] ):
							os.makedirs(KNN_output_folder +  img_name_split[0] + '/' + image[:-4])

						cv.imwrite(KNN_output_folder +  img_name_split[0] + '/' + image[:-4] + '/' + image[:-4] + '.' + str(i) + '.jpg', ROI_imgs[i])

				# image counter in the class imcrementing
				counter += 1


				if fold == 1 :
					if counter == 3:
						output_folder = f'datasets/dataset2/fold {fold}/orig_images/training/'
						ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/training/'

					if counter == 15:
						output_folder = f'datasets/dataset2/fold {fold}/orig_images/testing/'
						ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/testing/'
						counter = 0
				else:
					if counter == fold_counter:
						output_folder = f'datasets/dataset2/fold {fold}/orig_images/testing/'
						ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/testing/'

					if counter == fold_counter + 3:
						output_folder = f'datasets/dataset2/fold {fold}/orig_images/training/'
						ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/training/'

					if counter == 15:
						output_folder = f'datasets/dataset2/fold {fold}/orig_images/training/'
						ROI_output_folder = f'datasets/dataset2/fold {fold}/ROI_images/ROI_dataset/training/'
						counter = 0

			if fold != 1:
				fold_counter += 3
	
	def get_image_class(self, img_name):
		'''
		this function is only used for the third dataset : AR Faces
		it takes an image and return its class
		'''
		file_split = img_name.split('-')
		img_class = file_split[0]
		return img_class

	def prepare_dataset3(self, dataset_path):
		
		os.mkdir('datasets/dataset3')

		#----------------------------------------------------------- splitting the original dataset into train and test images------------------------------------
		# creating output folders
		os.mkdir('datasets/dataset3/orig_images')
		os.mkdir('datasets/dataset3/orig_images/training')
		os.mkdir('datasets/dataset3/orig_images/testing')

		os.mkdir('datasets/dataset3/ROI_images')
		# creating region of interest dataset folders
		
		os.mkdir('datasets/dataset3/ROI_images/ROI_dataset')
		os.mkdir('datasets/dataset3/ROI_images/ROI_dataset/training')
		os.mkdir('datasets/dataset3/ROI_images/ROI_dataset/testing')

		# creating KNN dataset
		os.mkdir('datasets/dataset3/ROI_images/KNN_dataset')

		# variable that stores the current output folder (either training or testing), for the original images and the ROI images
		output_folder = 'datasets/dataset3/orig_images/training/'
		ROI_output_folder = 'datasets/dataset3/ROI_images/ROI_dataset/training/'
		KNN_output_folder = 'datasets/dataset3/ROI_images/KNN_dataset/'


		# listing images in the dataset path
		img_list = np.array(os.listdir(dataset_path))

		counter = 0

		# for each image in the original dataset
		for image in img_list:
        	
        	# getting the image class
			folder = self.get_image_class(image)

			# creating the output classes folders
			if not path.exists(output_folder + folder):
				os.mkdir(output_folder + folder)

			# reading then saving the image in its corresponding folder
			img = cv.imread(dataset_path + image)
			cv.imwrite(output_folder + folder + '/' + image[:-4] + '.jpg', img)
        
			# create the output class folder if it doesn't exist for the region of interests
			if not path.exists(ROI_output_folder + folder):
				os.mkdir(ROI_output_folder + folder)

			# getting its regions of interest of the images 
			ROI_imgs = self.get_region_of_interest(img)

			# save each region of interest in its corresponding folder
			for i in range(len(ROI_imgs)):
				cv.imwrite(ROI_output_folder + folder + '/' +  image[:-4] + '_' + str(i) + '.jpg', ROI_imgs[i])

				# creating a folder for region of interest in the testing images (used for the KNN dataset)
				if 'testing' in ROI_output_folder:
					# creating the class folder with only the second part of the file_split : 8.jpg
					if not path.exists(KNN_output_folder +  folder + '/' + image[:-4] ):
						os.makedirs(KNN_output_folder +  folder + '/' + image[:-4])

					cv.imwrite(KNN_output_folder +  folder + '/' + image[:-4] + '/' + image[:-4] + '.' + str(i) + '.jpg', ROI_imgs[i])

			# changing the output directly based on : if the faces are covered : put the image in the test, else put it in the train
			if counter == 6:
				output_folder = 'datasets/dataset3/orig_images/testing/'
				ROI_output_folder = 'datasets/dataset3/ROI_images/ROI_dataset/testing/'

			if counter == 12:
				output_folder = 'datasets/dataset3/orig_images/training/'
				ROI_output_folder = 'datasets/dataset3/ROI_images/ROI_dataset/training/'

			if counter == 19:
				output_folder = 'datasets/dataset3/orig_images/testing/'
				ROI_output_folder = 'datasets/dataset3/ROI_images/ROI_dataset/testing/'

			counter += 1

			if counter == 26:
				output_folder = 'datasets/dataset3/orig_images/training/'
				ROI_output_folder = 'datasets/dataset3/ROI_images/ROI_dataset/training/'
				counter = 0




