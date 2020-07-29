import matplotlib
from keras import Sequential, Input, Model
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Bidirectional, Dense, Activation, Dropout
from keras import layers
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.layers import LSTM

matplotlib.use('TkAgg')

catergories = ['alt.atheism', 'sci.space']
df = fetch_20newsgroups(subset='train', shuffle=False, categories=catergories)
sentences = df.data
y = df.target

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

max_words = 100
max_len = 100
embdim = 100

tokenizer = Tokenizer(num_words=max_words)

max_review_len = max([len(s.split()) for s in sentences])
vocab_size = len(tokenizer.word_index) + 1
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
padded_train = pad_sequences(X_train_tokens, maxlen=max_review_len)
paded_test = pad_sequences(X_test_tokens, maxlen=100)

tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)


def LSTM_model():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, embdim, input_length=paded_test.shape[1])(inputs)
    layer = LSTM(64, activation='tanh', recurrent_activation='hard_sigmoid', dropout=.25, recurrent_dropout=.25)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


model = LSTM_model()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(sequences_matrix, y_train, batch_size=50000, epochs=5,
                    validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

test_loss, test_acc = model.evaluate(paded_test, y_test)
print(test_acc)

[test_loss, test_acc] = model.evaluate(paded_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# list all data in history
print(history.history.keys())
# summarize history for accuracy

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
