from statistics import mode
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU, Activation
from keras.layers.normalization import batch_normalization
from keras import optimizers
import tensorflow as tf
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D, AveragePooling1D
from prepatation_data import x_train

model = Sequential()

model.add(Convolution1D(32, 3, padding="same", activation="relu", input_shape=(x_train.shape[1],1)))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Dropout(0.5))

model.add(Convolution1D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(2, activation="softmax"))