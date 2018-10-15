import csv
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K

d180 = list(csv.reader(open('o2_d180.csv')))
age = list(csv.reader(open('o2_age.csv')))

k = 3
X, Y = [], []
for i in range(len(d180) - k):
    X.append(d180[i:i+k])
    Y.append(d180[i+k])
X = np.asanyarray(X)
Y = np.asanyarray(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

x_train = x_train.reshape(len(x_train), k, )
x_test = x_test.reshape(len(x_test), k, )
input_shape = (k,)

model = Sequential()
model.add(Dense(32, activation='tanh', input_shape=input_shape))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()
epochs = 10
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
