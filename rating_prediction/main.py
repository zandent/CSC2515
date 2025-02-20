import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K
import keras.layers
import tensorflow as tf
from keras.callbacks import EarlyStopping

import json
import re
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt

def list_splitter(list_to_split, ratio):
    first_half = int(len(list_to_split) * ratio)
    return list_to_split[:first_half], list_to_split[first_half:]
def Convert(tup, di): 
    for a, b in tup: 
        di.setdefault(a, b)
    return di

# y_pred = numpy.array([4.8483334e-06, 1.4581978e-03, 2.2443533e-03, 1.3272554e-02, 1.3884255e-01,7.5374603e-01])
# print (str(numpy.argmax(y_pred)))

# fix random seed for reproducibility
numpy.random.seed(7)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 20000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# print(len(X_train[0]))
# print(len(X_train[578]))
# print(y_train.shape)
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# print(X_train)
# print(y_train)
# print(X_train.shape)
# print(y_train.shape)

X_train = []
y_train = []
wordcount = defaultdict(list)
i = 0
with open('data/train.json', 'r') as train_file:
    for row in train_file:
        data = json.loads(row)
        if "reviewText" in data:
            i += 1
            a_string = data['reviewText']
            a_list = a_string.split()
            # print (len(a_list))
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm != '':
                    if s_rm in wordcount:
                        wordcount[s_rm] += 1
                    else:
                        wordcount[s_rm] = 1
        if "summary" in data:
            sa_string = data['summary']
            sa_list = sa_string.split()
            for s in sa_list:
                ss_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if ss_rm != '':
                    if ss_rm in wordcount:
                        wordcount[ss_rm] += 2
                    else:
                        wordcount[ss_rm] = 2
skip_word = 22
dic_list = sorted(wordcount.items(), key=lambda item: item[1], reverse=True)
dic_list = dic_list[:top_words]
for j in range (len(dic_list)):
    dic_list[j] = (dic_list[j][0],j)
wordcount = defaultdict(list)
Convert(dic_list, wordcount)
# print (wordcount)
i = 0
with open('data/train.json', 'r') as train_file:
    for row in train_file:
        data = json.loads(row)
        r = float(data['overall'])
        post_list = []
        if "reviewText" in data:
            i += 1
            a_string = data['reviewText']
            a_list = a_string.split()
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm in wordcount:
                    indices = wordcount[s_rm]
                    post_list.append(indices)
        if "summary" in data:
            a_string = data['summary']
            a_list = a_string.split()
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm in wordcount:
                    indices = wordcount[s_rm]
                    post_list.append(indices)
        if post_list:    
            X_train.append(post_list)
            y_train.append(r)

X_train = numpy.array(X_train)
y_train = numpy.array(y_train)
# X_cmp = X_train
# y_cmp = y_train
# idx = numpy.random.permutation(len(X_train))
# X_train,y_train = X_train[idx], y_train[idx]
# print ("Xtrain is", X_train[100])
# print ("ytrain is", y_train[100])
# print ("idx is", idx[100])
# print ("Xcmp is", X_cmp[idx[100]])
# print ("ycmp is", y_cmp[idx[100]])
(_, X_test) = list_splitter(X_train, 0.8)
(_, y_test) = list_splitter(y_train, 0.8)
# (X_train, X_test) = list_splitter(X_train, 0.8)
# (y_train, y_test) = list_splitter(y_train, 0.8)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# print(X_train)
# print(y_train)
# print(X_train.shape)
# print(y_train.shape)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, embeddings_initializer = 'zeros',input_length=max_review_length))
model.add(GRU(32))
model.add(Dropout(0.1))
# model.add(Dense(250, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# simple early stopping, optional
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit(X_train, y_train, epochs=5, batch_size=64,validation_data = (X_test,y_test),callbacks=[es])
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('gru_accuracy.pdf')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('gru_loss.pdf')
plt.close()
X_pred = []
i = 0
with open('data/test.json', 'r') as train_file:
    for row in train_file:
        data = json.loads(row)
        # if "reviewText" in data:
        i += 1
        post_list = []
        if "reviewText" in data:
            a_string = data['reviewText']
            a_list = a_string.split()
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm in wordcount:
                    indices = wordcount[s_rm]
                    post_list.append(indices)
        if "summary" in data:
            a_string = data['summary']
            a_list = a_string.split()
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm in wordcount:
                    indices = wordcount[s_rm]
                    post_list.append(indices)
        if not "reviewText" in data and not "summary" in data:
            print ("CANNOT PREDICT THIS DATA: ",data)
        if post_list:
            X_pred.append(post_list)
        else:
            X_pred.append([0])
X_pred = numpy.array(X_pred)
print ("x_pred_prev shape", X_pred.shape)
X_pred = sequence.pad_sequences(X_pred, maxlen=max_review_length)
y_pred = model.predict(X_pred)
print ("X_pred_late shape", X_pred.shape)
print ("y_pred_late shape", y_pred.shape)
i = 0
predictions = open('submission.csv', 'w')
for l in open('data/rating_pairs.csv'):
    if l.startswith('userID'):
        predictions.write(l)
        continue
    u,p = l.strip().split('-')
    # print ("y_pred", y_pred[i], " and ", i)
    predictions.write(u + '-' + p + ',' + str(numpy.argmax(y_pred[i])) + '.0\n')
    i += 1
predictions = open('rating_predictions_bak.csv', 'w')
for j in y_pred:
    predictions.write(str(numpy.argmax(j)) + '\n')