from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


class CnnClassifier:
    def build_cnn_model(self, nb_classes, input_shape=32):
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
        classifier.add(Dropout(0.2))
        classifier.add(Dense(activation="softmax", units=nb_classes))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return classifier

    def model_train(self, nb_classes, batch_size, input_size, train_path, test_path):
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
        self.model_train(nb_classes=40, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)

    def dataset2(self, train_path, test_path, batch_size, input_size):
        self.model_train(nb_classes=50, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)

    def dataset3(self, train_path, test_path, batch_size, input_size):
        self.model_train(nb_classes=100, batch_size=batch_size, input_size=input_size, train_path=train_path, test_path=test_path)


# -------------------------------------------Main------------------------------------------------
# creating an object from the cnn module class
classifier_obj = CnnClassifier()

# # AT&T dataset
dataset1_train_path = "../../datasets/dataset1/orig_images/training"
dataset1_test_path = "../../datasets/dataset1/orig_images/testing"
classifier_obj.dataset1(train_path=dataset1_train_path, test_path=dataset1_test_path, batch_size=32, input_size=32)

# Georgia Tech dataset
dataset2_train_path = "../../datasets/dataset2/fold 1/orig_images/training"
dataset2_test_path = "../../datasets/dataset2/fold 1/orig_images/testing"
# classifier_obj.dataset2(train_path=dataset2_train_path, test_path=dataset2_test_path, batch_size=32, input_size=32)

# AR Faces dataset
dataset3_train_path = "../../datasets/dataset3/orig_images/training"
dataset3_test_path = "../../datasets/dataset3/orig_images/testing"
# classifier_obj.dataset3(train_path=dataset3_train_path, test_path=dataset3_test_path, batch_size=32, input_size=32)