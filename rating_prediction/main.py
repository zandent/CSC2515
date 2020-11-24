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

# fix random seed for reproducibility
numpy.random.seed(7)
tf.random.set_seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 70000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# print(len(X_train[0]))
# print(len(X_train[578]))
# print(y_train.shape)
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)

X_train = []
y_train = []
userRatings = defaultdict(list)
i = 0

with open('data/train.json', 'r') as train_file:
    for row in train_file:
        data = json.loads(row)
        # print (data)
        r = float(data['overall'])
        if "reviewText" in data:
            i += 1
            a_string = data['reviewText']
            a_list = a_string.split()
            post_list = []
            for s in a_list:
                s_rm = re.sub(r'[^A-Za-z]', '', s).lower()
                if s_rm != '':
                    hash_object = hashlib.sha256(s_rm.encode())
                    hex_dig = int(hash_object.hexdigest()[:4], 16)
                    post_list.append(hex_dig)
            X_train.append(post_list)
            y_train.append(r)
            # if i <= 2:
            #     print (post_list)
            #     print ("X_train is ", X_train)
            # +data['summary'])
X_train = numpy.array(X_train)
y_train = numpy.array(y_train)
(X_train, X_test) = list_splitter(X_train, 0.8)
(y_train, y_test) = list_splitter(y_train, 0.8)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, embeddings_initializer = 'zeros',input_length=max_review_length))
# 4 different cells: default rnncell/GRU/LSTM/self-defined rnncell
# model.add(RNN(SimpleRNNCell(32)))
# model.add(RNN(MinimalRNNCell(32)))
model.add(GRU(32))
# model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# simple early stopping, optional
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(X_train, y_train, epochs=10, batch_size=64,validation_data = [X_test,y_test],callbacks=[es])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
