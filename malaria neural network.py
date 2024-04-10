'''
About Dataset
Content
The dataset contains 2 folders

Infected
Uninfected
And a total of 27,558 images.
Acknowledgements
This Dataset is taken from the official NIH Website:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/
And uploaded here, so anybody trying to start working with this dataset can get started immediately,
as to download the
dataset from NIH website is quite slow.
Photo by Егор Камелев on Unsplash
https://unsplash.com/@ekamelev

Inspiration
Save humans by detecting and deploying Image Cells that contain Malaria or not!

dataset took from:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
'''

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Normalization, Rescaling, RandomFlip, RandomRotation, Dropout
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping

im_height = 64
im_width = 64
train_path = 'C:/Users/User01/Desktop/malaria image dataset/Train/train'
val_path = 'C:/Users/User01/Desktop/malaria image dataset/Train/val'
test_path = 'C:/Users/User01/Desktop/malaria image dataset/Test'

train_dataset = image_dataset_from_directory(train_path,
                                             image_size = (im_width,im_height),
                                             seed = 42,
                                             shuffle = True,
                                             batch_size = 128,
                                             label_mode = 'binary'
                                             )
val_dataset = image_dataset_from_directory(val_path,
                                           image_size = (im_width, im_height),
                                           seed = 42,
                                           shuffle = True,
                                           label_mode = 'binary',
                                           batch_size = 128
                                           )
test_dataset = image_dataset_from_directory(test_path,
                                            image_size = (im_width, im_height),
                                            label_mode = 'binary',
                                            shuffle = True,
                                            seed = 42,
                                            batch_size = 128
                                            )

stop = EarlyStopping(min_delta = 0.0005,
                     patience = 3,
                     restore_best_weights = True)

model = Sequential([
    #Normalization(),
    Rescaling(1/255), # you can use Normalization or Rescaling, the latter is slighlty better
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.2),
    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    MaxPool2D(),
    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    MaxPool2D(),
    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
    MaxPool2D(),
    Flatten(),
    Dense(units = 128, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 128, activation = 'relu'),
    Dropout(0.5),
    Dense(units = 1, activation = 'sigmoid')

])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])

training = model.fit(train_dataset,
            validation_data = val_dataset,
            epochs = 50,
            callbacks = [stop])

history = pd.DataFrame(training.history)
history.plot()
plt.grid()
plt.show()
print('TEST')
model.evaluate(test_dataset)
model.save('malaria neural network')