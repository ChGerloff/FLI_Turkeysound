#!/usr/bin/env python
# coding: utf-8

# # More Streamlined version of the first CNN, is basically the out of the box standard CNN 

# ## Load all Packages (Not all needed)

# In[1]:
import tensorflow as tf

import os
import glob
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.signal
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import IPython.display as ipd
import time
from glob import glob
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from skimage.io import imread_collection
import datetime
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

from PIL import Image
import os, os.path

import cv2
import glob


# ## Define Batch size and size of Input Data

batch_size = 32
img_height = 220
img_width = 338


# ## Load the Training and the Validation Data, the files are split into folders in the directory, representing the different categories
# ### In this the folders are split into Sick/Healthy

import random
random.seed(222)
data_dir = '/data/users/christoph.gerloff/KT_4/'



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  #shuffle=true,
  image_size=(img_height, img_width),
  batch_size=batch_size)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  #shuffle=true,
  image_size=(img_height, img_width),
  batch_size=batch_size)



# ## Define and initialize the model, added rescaling and dropout layers


model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
#  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
#  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
#  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, kernel_regularizer='l1_l2', 
                        activation='relu'),
  #tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
  optimizer=opt,
  #optimizer=RMSprop(lr=0.001),
  loss='binary_crossentropy',
  metrics=['accuracy'])


# # Train the Model:

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100#,
  #callbacks=[tensorboard_callback]
)

scores = model.evaluate(train_ds, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# serialize model to JSON
model_json = model.to_json()
with open("/data/users/christoph.gerloff/Plots/model4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/data/users/christoph.gerloff/Plots/model_4splits.h5")
print("Saved model to disk")


from sklearn.metrics import confusion_matrix
y_true = []
y_pred = []
for x,y in val_ds:
  y_true.append(y)
  y_pred.append((model.predict(x)> 0.5).astype("int32"))

    
        
y_pred = tf.concat(y_pred, axis=0)
y_true = tf.concat(y_true, axis=0)
confusion_mtx = confusion_matrix(y_true, y_pred)



print(y_pred)


# ## Make a numpy array out of the input categories (i think not needed anymore)
import seaborn as sns

confusion_mtx = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True,
            fmt="d"
           );
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/data/users/christoph.gerloff/Plots/Confusion_4split.pdf')
plt.close()

# ## Save the Model weights


with open('/data/users/christoph.gerloff/Plots/modelsummary_4split.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.summary()


# ## Plot the Accuracy and the Loss Function

print("Plotting accuracy versus epoch")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
print("The model is being evaluated")
#test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)
#print("The accuracy of the model is:")
#print(test_acc)
plt.savefig('/data/users/christoph.gerloff/Plots/Acc_4split.pdf')
plt.close()

import matplotlib.pyplot as plt
loss = history.history['loss']
#loss = tf.Variable(loss)
val_loss = history.history['val_loss']
#val_loss = tf.Variable(val_loss)
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/data/users/christoph.gerloff/Plots/Loss_4split.pdf')
plt.show()
plt.close()

# ## Initialize the confusion Matrix for the test weeks

# ### Load in the rest data


data_dir_test = '/data/users/christoph.gerloff/KT_4_Test'
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
  #validation_split=0.2,
  #subset="validation",
  seed=123,
  #shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ### Create and define the needed variables and objects

from sklearn.metrics import confusion_matrix
y_true_test = []
y_pred_test = []
for x,y in test_ds:
  y_true_test.append(y)
  y_pred_test.append((model.predict(x)> 0.5).astype("int32"))

    
        
y_pred_test = tf.concat(y_pred_test, axis=0)
y_true_test = tf.concat(y_true_test, axis=0)

# ### Plot the Validation Confusion Matrix

import seaborn as sns

confusion_mtx_test = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx_test, annot=True,
            fmt="d"
           );
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/data/users/christoph.gerloff/Plots/Confusion_test_4split.pdf')
plt.close()