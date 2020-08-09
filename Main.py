from preprocessing import Preprocessing
from Our_proposed_method.Python_files.CNN_ROI import CnnROIModule
from Our_proposed_method.Python_files.Decision_Module import decisionModule
from keras.models import load_model
from os import path
import os


def run_first_module():
	'''
	function that execues the first module which prepare the all 3 datasets (creates the datasets splits and creates the ROI and KNN datasets)
	'''
	if not path.exists('datasets'):	
		# creating a dataset folder to store all 3 datasets 
		os.mkdir('datasets')

		print('The preparation could take about 2 minutes please wait')

		# instanciating the first module 
		proceprocessor = Preprocessing()

		# path for the first dataset (AT&T dataset)
		dataset1_path = 'original datasets/AT&T/'

		# preparing the first dataset
		proceprocessor.prepare_dataset1(dataset_path=dataset1_path)
		print('AT&T dataset has been successfully prepared')

		# path for the second dataset (Georgia Tech dataset)
		dataset2_path = 'original datasets/Georgia Tech/'

		# preparing the second dataset
		proceprocessor.prepare_dataset2(dataset_path=dataset2_path)
		print('Georgia Tech dataset has been successfully prepared')

		# path for the third dataset (AR Faces dataset)
		dataset3_path = 'original datasets/AR Faces/'

		# preparing the second dataset
		proceprocessor.prepare_dataset3(dataset_path=dataset3_path)
		print('AR Faces dataset has been successfully prepared')

	else: 
		print('The datasets are ready')


def run_second_module(picked_dataset, batch_size, input_size):
	'''
	function that execues the second module which is the CNN, which classifies the ROI images and saves the model (with the best weights)
	parameteres:
	picked_dataset: the dataset on which the CNN applies on 
	batch_size : the batch size of the CNN model
	input_size : the input size of the images
	'''
	# creating an object from the cnn module class
	CNN_module_obj = CnnROIModule()

	if picked_dataset == 1:
		dataset1_train_path = "datasets/dataset1/ROI_images/ROI_dataset/training"
		dataset1_test_path = "datasets/dataset1/ROI_images/ROI_dataset/testing"
		CNN_module_obj.dataset1(train_path=dataset1_train_path, test_path=dataset1_test_path, batch_size=batch_size, input_size=input_size)

	elif picked_dataset == 2:

		dataset2_train_path = "datasets/dataset2/fold 5/ROI_images/ROI_dataset/training"
		dataset2_test_path = "datasets/dataset2/fold 5/ROI_images/ROI_dataset/testing"
		CNN_module_obj.dataset2(train_path=dataset2_train_path, test_path=dataset2_test_path, batch_size=batch_size, input_size=input_size)

	else:
		# AR Faces dataset
		dataset3_train_path = "datasets/dataset3/ROI_images/ROI_dataset/training"
		dataset3_test_path = "datasets/dataset3/ROI_images/ROI_dataset/testing"

		CNN_module_obj.dataset3(train_path=dataset3_train_path, test_path=dataset3_test_path, batch_size=batch_size, input_size=input_size)


def run_third_module(picked_dataset):
	'''
	function that execues the third module which applies the final classification on the specified dataset and print the accuracy
	parameters: 
	'''

	# loading the model with the best weights
	classifier = load_model('mode_bestweights.hdf5')
	# variable only used for the second dataset
	fold = 5  

	if picked_dataset == 1:
		# AT&T dataset
		dataset1_knn_test_path = "datasets/dataset1/ROI_images/KNN_dataset/"
		decisionModule.Calculate_model_accuracy(classifier=classifier, dataset_path=dataset1_knn_test_path, nbr_classes=40, tst_img_per_class=3)

	elif picked_dataset == 2:
		# Georgia Tech dataset
		dataset2_knn_test_path = f"datasets/dataset2/fold {fold}/ROI_images/KNN_dataset/"
		decisionModule.Calculate_model_accuracy(classifier=classifier, dataset_path=dataset2_knn_test_path, nbr_classes=50, tst_img_per_class=3)

	else:
		# AR Faces dataset
		dataset3_knn_test_path = "datasets/dataset3/ROI_images/KNN_dataset/"
		decisionModule.Calculate_model_accuracy(classifier=classifier, dataset_path=dataset3_knn_test_path, nbr_classes=100, tst_img_per_class=12)


# Main ---------------------
# the first module is only executed once  
# then run the second and third module - the dataset need to be specified (as a parameters) for these two modules
dataset = 1

run_first_module()     

run_second_module(picked_dataset=dataset, batch_size=32, input_size=32)

run_third_module(dataset)


