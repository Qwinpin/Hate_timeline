from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential, Model
from pandas import read_csv
from keras.preprocessing import sequence
import numpy as np
from keras.preprocessing.text import Tokenizer

sequence_length = 2190
encoded_legth = 30

dataset = read_csv('.//dataset_timeseris.csv').iloc[:,1:].fillna(0)

temp = np.array(dataset.values.tolist())
temp2 = []

for row in temp:
    new = np.split(row, 30)
    temp2.append(new)

x_train = np.array(temp2)
x_train = sequence.pad_sequences(x_train)

x_test = x_train[len(x_train)//10*8:]
x_train = x_train[:len(x_train)//10*8]

timesteps, input_dim = 30, 73

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(10)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(loss='binary_crossentropy',
                        optimizer='RMSprop',
                        metrics=['accuracy'])

sequence_autoencoder.fit(x_train, x_train,
                        batch_size=1,
                        epochs=10,
                        validation_data=(x_test, x_test))

res = encoder.predict(x_test)
