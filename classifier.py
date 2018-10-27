import spacy
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import numpy as np
import json 
from spacy.lang.en.stop_words import STOP_WORDS
import string

def get_doc_vector(ipt):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(ipt)
    return doc.vector

labels = ['morning','evening', 'supper', 'breakfast']

with open('train_data.json', 'r') as train_data:
    data = json.load(train_data)


vector_map = {}
for key,values in data.items():
    for value in values.split(','):
        label = labels.index(value) 
        vector_map[label] = get_doc_vector(key)

model = keras.Sequential()

model.add(Input(shape=(4,), dtype='float32'))
#model.add(keras.layers.Embedding(len(vector_map), 16))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_train = []
y_train = []
for key,value in vector_map.items():
    x_train.append(value)
    y_train.append(key)

x_val = x_train[2:]
y_val = y_train[2:]
x_train = x_train[2:]
y_train = y_train[2:]
history = model.fit(x_train,
                    y_train,
                    epochs=40,
                    validation_data=(x_val, y_val),
                    verbose=1)