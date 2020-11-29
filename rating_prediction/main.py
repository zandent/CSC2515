# import graphlab
# from graphlab import SFrame
# product_reviews=graphlab.SFrame.read_json('data/train.json', orient='lines')
# # print (product_reviews.head())
# # product_reviews.print_rows()
# product_reviews['overall'] = [int(i) for i in product_reviews['overall']]
# product_reviews['wordcount'] = graphlab.text_analytics.count_words(product_reviews['reviewText']+product_reviews['summary'])
# vs_reviews = graphlab.SFrame.read_json('data/test.json', orient='lines')
# vs_reviews['wordcount'] = graphlab.text_analytics.count_words(vs_reviews['reviewText']+vs_reviews['summary'])
# vs_reviews.print_rows()

# # product_reviews = product_reviews[product_reviews['overall'] !=3]
# # product_reviews['posrating'] = product_reviews['overall'] >= 4
# product_reviews = product_reviews.dropna()
# train_data, test_data = product_reviews.random_split(0.8, seed=0)
# sentiment_model = graphlab.logistic_classifier.create (train_data, target='overall', features=['wordcount'],validation_set=test_data, max_iterations=100)
# sentiment_model.evaluate(test_data, metric='roc_curve')
# # print sentiment_model


# results=sentiment_model.predict_topk(test_data, output_type='probability', k=1)
# results['class'] = [float(i) for i in results['class']]
# test_data.add_columns(results)
# test_data=test_data.sort('overall', ascending=False)
# # print results
# print test_data
# print test_data.tail()

# # results=sentiment_model.predict_topk(vs_reviews, output_type='probability', k=1)
# # results['class'] = [float(i) for i in results['class']]
# # vs_reviews.add_columns(results)
# # vs_reviews=vs_reviews.sort('class', ascending=False)
# # # print results
# # print vs_reviews
# # print vs_reviews.tail()

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM,RNN,LSTMCell,SimpleRNNCell,GRU
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
class MinimalRNNCell(keras.layers.Layer):
    """
    self defined RNNCELL, mindstorm
    """

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
        #                               initializer='uniform',
        #                               name='kernel')
        # self.recurrent_kernel = self.add_weight(
        #     shape=(self.units, self.units),
        #     initializer='uniform',
        #     name='recurrent_kernel')
        self.b = self.add_weight(shape = (self.units,),initializer='uniform')
        self.b2 = self.add_weight(shape = (self.units,),initializer='uniform')
        # self.b3 = self.add_weight(shape = (self.units,),initializer='uniform')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]

        # h = K.dot(inputs, self.kernel)
        # output = h + K.dot(prev_output, self.recurrent_kernel)+self.b
        # if K.l2_normalize(prev_output)<0.000001:
        # 	return inputs,[inputs]
        # if K.l2_normalize(inputs)<0.000001:
        # 	return prev_output,[prev_output]
        output = K.reshape(K.batch_dot(K.reshape(self.b2+prev_output,shape=(-1,side,side)),K.reshape(self.b+inputs,shape=(-1,side,side))),shape=(-1,side**2))
        print("here: ", prev_output,inputs,output)

        return output, [output]

class myRNN(RNN):
    def get_initial_state(self,inputs):
    	i = tf.broadcast_to(K.reshape(K.eye(side),shape=(1,side*side)),[K.shape(inputs)[0],side*side])
    	print("here\n\n\n",i)
    	print("")
    	return [i]
		# return tf.ones((batch_size, self.state_size))

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
# 4 different cells: default rnncell/GRU/LSTM/self-defined rnncell
# model.add(RNN(SimpleRNNCell(32)))
#model.add(RNN(MinimalRNNCell(32)))
model.add(Bidirectional(GRU(32)))
# model.add(LSTM(32))
# model.add(Dropout(0.1))
# model.add(Dense(250, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# simple early stopping, optional
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(X_train, y_train, epochs=3, batch_size=64,validation_data = (X_test,y_test))
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

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
predictions = open('rating_predictions.csv', 'w')
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