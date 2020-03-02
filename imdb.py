#! /usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import numpy as numpy

data = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

# preprocessing
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

print(decode_review(test_data[0]))

model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16)) # create 10,000 word vectors of size 16
model.add(keras.layers.GlobalAveragePooling1D()) # decreases the dimensions of each vector using averages
model.add(keras.layers.Dense(16,activation='relu')) # feed into 16 neurons
model.add(keras.layers.Dense(1,activation='sigmoid')) # final output neuron, where a 0 or 1 decision is made

model.summary()

model.compile(
    optimizer = "adam",
    loss="binary_crossentropy",
    metrics = ["accuracy"]
)

validation_data, train_data = train_data[:10000] , train_data[10000:]
validation_labels, train_labels = train_labels[:10000] , train_labels[10000:]

fit = model.fit(train_data,train_labels,epochs=40,batch_size=512,validation_data=(validation_data,validation_labels),verbose=1)

results = model.evaluate(test_data,test_labels)

print(results)
predictions = model.predict(test_data)
for i,prediction in enumerate(predictions):
    print("Review:\n",decode_review(test_data[i]))
    print("prediction: ",str(prediction))
    print("Actual: ",str(test_labels[i]))
    if i==50:
        break
model.save("model.h5")
