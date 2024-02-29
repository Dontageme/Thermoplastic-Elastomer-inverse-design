import os
import numpy as np

# Model Part
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,Input,Normalization
from tensorflow.keras.callbacks import ModelCheckpoint

#Model
model = keras.Sequential()
model.add(Input(shape=(3)))
model.add(Normalization(axis=None))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(101))
optimizer=tf.keras.optimizers.Adam(0.001)
checkpointer = ModelCheckpoint(filepath='polymer.hdf5',save_best_only=True)
model.compile(loss="mean_squared_error",optimizer=optimizer,metrics=["mean_squared_error"])
model.summary()
