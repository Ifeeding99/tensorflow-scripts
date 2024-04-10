import tensorflow as tf
import os
from PIL import Image
import numpy as np
import tqdm
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Dense

print(tf.__version__)



train_path = "C:/Users/flavi/Downloads/carvana-image-masking-challenge/train"
mask_path = "C:/Users/flavi/Downloads/carvana-image-masking-challenge/train_masks/train_masks"

train_image_list = [os.path.join(train_path,p) for p in os.listdir(train_path)]
mask_list = [os.path.join(mask_path,p) for p in os.listdir(mask_path)]

'''loop = tqdm.tqdm(train_image_list)
for i,image in enumerate(loop):
    im = np.asarray(Image.open(os.path.join(train_path, image)))
    im = im/255
    m = np.asarray(Image.open(os.path.join(mask_path,mask_list[i])))
    m = m/255
    images = np.append(images, im)
    masks = np.append(masks, m)
    
    loop.set_description('Caricando le immagini')'''

def decode(x,y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    y = tf.io.read_file(y)
    y = tf.image.decode_gif(y)
    y = y[:, :, 0]
    y = tf.expand_dims(y, axis=-1)
    return x,y
    _

dataset = tf.data.Dataset.from_tensor_slices((train_image_list,mask_list))
dataset.map(decode)

input_ = Input(shape=(None,None,3))
x = tf.keras.layers.Rescaling(1/255)(input_)
x = Conv2D(64,(3,3),(1,1),'same', activation='relu')(x)
x_1 = Conv2D(64,(3,3),(1,1),'same', activation='relu')(x)
x = MaxPool2D((2,2),(2,2))(x_1)
x = Conv2D(128,(3,3),(1,1),'same', activation='relu')(x)
x_2 = Conv2D(128,(3,3),(1,1),'same', activation='relu')(x)
x = MaxPool2D((2,2),(2,2))(x_2)
x = Conv2D(256,(3,3),(1,1),'same', activation='relu')(x)
x_3 = Conv2D(256,(3,3),(1,1),'same', activation='relu')(x)
x = MaxPool2D((2,2),(2,2))(x_3)
x = Conv2D(512,(3,3),(1,1),'same', activation='relu')(x)
x_4 = Conv2D(512,(3,3),(1,1),'same', activation='relu')(x)
x = MaxPool2D((2,2),(2,2))(x_4)
conv_5 = Conv2D(1024,(3,3),(1,1),'same', activation='relu')(x)
conv_6 = Conv2D(1024,(3,3),(1,1),'same', activation='relu')(conv_5)
conv_up = Conv2DTranspose(512, (2,2), (2,2), activation='relu')(conv_6)
up1 = tf.keras.layers.Concatenate(axis=-1)([x_4, conv_up])
x = Conv2D(512,(3,3),(1,1),'same',activation='relu')(up1)
x = Conv2D(512,(3,3),(1,1),'same',activation='relu')(x)
conv_up_2 = Conv2DTranspose(256,(2,2),(2,2),activation='relu')(x)
up2 = tf.keras.layers.Concatenate(axis=-1)([x_3,conv_up_2])
x = Conv2D(256,(3,3),(1,1),'same',activation='relu')(up2)
x = Conv2D(256,(3,3),(1,1),'same',activation='relu')(x)
conv_up_3 = Conv2DTranspose(128,(2,2),(2,2),activation='relu')(x)
up3 = tf.keras.layers.Concatenate(axis=-1)([x_2,conv_up_3])
x = Conv2D(128,(3,3),(1,1),'same',activation='relu')(up3)
x = Conv2D(128,(3,3),(1,1),'same',activation='relu')(x)
conv_up_3 = Conv2DTranspose(64,(2,2),(2,2),activation='relu')(x)
up4 = tf.keras.layers.Concatenate(axis=-1)([x_1,conv_up_3])
x = Conv2D(64,(3,3),(1,1),'same',activation='relu')(up4)
x = Conv2D(64,(3,3),(1,1),'same',activation='relu')(x)
output = Conv2D(1,(1,1),(1,1),activation='sigmoid')(x)

model = Model(inputs=[input_], outputs = [output])

c = 0
for el in dataset:
    if c >= 1:
        break
    else:
        print(el)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, nesterov = True, momentum = 0.9),
              loss=bce_dice_loss,
              metrics = ['accuracy'], run_eagerly=True)
model.fit(dataset,
          epochs = 10)














