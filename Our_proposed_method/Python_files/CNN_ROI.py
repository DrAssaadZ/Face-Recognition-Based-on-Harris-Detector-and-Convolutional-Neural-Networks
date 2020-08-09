from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


class CnnROIModule:
    def build_cnn_model(self, nb_classes, input_shape=32):
        '''
        function that creates and returns the CNN model
        parameters: 
        nb_classes : the number of classes of the dataset
        input_shape : the input shape of the images
        '''
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), input_shape=(input_shape, input_shape, 3), padding='same', activation="relu"))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=512))
        classifier.add(Dense(activation="softmax", units=nb_classes))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return classifier

    def model_train(self, nb_classes, batch_size, input_size, train_path, test_path):
        '''
        function that train a CNN model on a given dataset, it doesn't return anything
        parameters:
        nb_classes: number of classes
        batch_size : size of the batch 
        input_size : input size of the images
        train_path : the train dataset path
        test_path : the test dataset path
        '''

        # initializing a train and a test data-generator
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # pre-processing for training set
        train_set = train_datagen.flow_from_directory(directory=train_path, target_size=(input_size, input_size),
                                                      batch_size=batch_size, class_mode='categorical')

        # pre-processing for test set
        test_set = test_datagen.flow_from_directory(directory=test_path, target_size=(input_size, input_size),
                                                    batch_size=batch_size, shuffle=False, class_mode='categorical')

        # saving the best weights of the model (where the validation accuracy is best)
        checkpoint = ModelCheckpoint(filepath="mode_bestweights.hdf5", monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')

        # building a model
        classifier = self.build_cnn_model(nb_classes=nb_classes, input_shape=input_size)

        # # training our model with 25 epochs
        classifier.fit_generator(train_set, steps_per_epoch=train_set.samples // batch_size + 1, epochs=25,
                                 callbacks=[checkpoint], validation_data=test_set,
                                 validation_steps=test_set.samples // batch_size + 1)

    def dataset1(self, train_path, test_path, batch_size, input_size):
        '''
        function that train the CNN model on the first dataset (AT&T)
        '''
        self.model_train(nb_classes=40, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)

    def dataset2(self, train_path, test_path, batch_size, input_size):
        '''
        function that train the CNN model on the second dataset (Georgia Tech)
        '''
        self.model_train(nb_classes=50, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)
        
    def dataset3(self, train_path, test_path, batch_size, input_size):
        '''
        function that train the CNN model on the third dataset (AR Faces)
        '''
        self.model_train(nb_classes=100, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)



