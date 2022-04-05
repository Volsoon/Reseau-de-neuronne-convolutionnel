# import modules
from keras.preprocessing.image import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

dataset_train_directorie = "dataset/training_set"
dataset_test_directorie = "dataset/test_set"
img_width = 128
img_height = 128

classifier = Sequential()

classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
 input_shape=(img_width, img_height, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=2))

classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1,
 activation="relu"))

classifier.add(MaxPooling2D(pool_size=2))

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
 activation="relu"))

classifier.add(MaxPooling2D(pool_size=2))

classifier.add(Convolution2D(filters=64, kernel_size=3, strides=1,
 activation="relu"))

classifier.add(MaxPooling2D(pool_size=2))


classifier.add(Flatten())

classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


train_data_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True,
 zoom_range=0.2, shear_range=0.2)

test_data_generator = ImageDataGenerator(rescale=1./255)

training_set = train_data_generator.flow_from_directory(
    dataset_train_directorie,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="binary"
)

test_set = test_data_generator.flow_from_directory(
    dataset_test_directorie,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="binary"
)

classifier.fit(
    training_set,
    steps_per_epoch=250,
    epochs=50,
    validation_data=test_set,
    validation_steps=63
)
"""
img = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg", target_size=(img_height, img_width))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)



result = classifier.predict(img)
print(result)
"""