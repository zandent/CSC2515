#######################################
### -------- Load libraries ------- ###

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split


#######################################
### --------- Import data --------- ###

# Import data from csv
data = pd.read_json('data/train.json', lines=True)

# Select required columns
data = data[['reviewText', 'summary', 'overall']]

# Remove a row if any of the three remaining columns are missing
data = data.dropna()

# Remove rows, where the label is present only ones (can't be split)
# data = data.groupby('overall').filter(lambda x : len(x) > 1)

# Set your model output as categorical and save in new label col
data['overall_label'] = pd.Categorical(data['overall'])

# Transform your output to numeric
data['overall'] = data['overall_label'].cat.codes

# Split into train and test - stratify over Issue
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['overall']])


#######################################
### --------- Setup BERT ---------- ###

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 100

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)


#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
overall = Dense(units=len(data.overall_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='overall')(pooled_output)
outputs = {'overall': overall}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()


#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'overall': CategoricalCrossentropy(from_logits = True)}
metric = {'overall': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
y_issue = to_categorical(data['overall'])

# Tokenize the input (takes some time)
x = tokenizer(
    text=(data['reviewText']+data['summary']).to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# Fit the model
history = model.fit(
    # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    x={'input_ids': x['input_ids']},
    y={'overall': y_issue},
    validation_split=0.2,
    batch_size=64,
    epochs=10)


#######################################
### ----- Evaluate the model ------ ###

# Ready test data
test_y_issue = to_categorical(data_test['overall'])
test_x = tokenizer(
    text=(data['reviewText']+data['summary']).to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

# Run evaluation
model_eval = model.evaluate(
    x={'input_ids': test_x['input_ids']},
    y={'overall': test_y_issue}
)
# import numpy
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping

# import json
# import re
# from collections import defaultdict

# from transformers import TFBertModel,  BertConfig, BertTokenizerFast

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# # Name of the BERT model to use
# model_name = 'bert-base-uncased'

# # Load transformers config and set output_hidden_states to False
# config = BertConfig.from_pretrained(model_name)
# config.output_hidden_states = False

# # Load BERT tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# # Load the Transformers BERT model
# model = TFBertModel.from_pretrained(model_name, config = config)
# max_review_length = 500
# def map_example_to_dict(input_ids, attention_masks, token_type_ids):
#   return {
#       "input_ids": input_ids,
#       "token_type_ids": token_type_ids,
#       "attention_mask": attention_masks,
#   }
# def encode_examples(X):
#     # prepare list, so that we can build up final TensorFlow dataset from slices.
#     input_ids_list = []
#     token_type_ids_list = []
#     attention_mask_list = []
#     for s in X:
#         bert_input = tokenizer.encode_plus(text=s,add_special_tokens = True, max_length = max_review_length, pad_to_max_length = True, return_attention_mask = True, truncation=True)
#         input_ids_list.append(bert_input['input_ids'])
#         token_type_ids_list.append(bert_input['token_type_ids'])
#         attention_mask_list.append(bert_input['attention_mask'])
#     return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list)).map(map_example_to_dict)
# # map to the expected input to TFBertForSequenceClassification, see here 
# def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
#   return {
#       "input_ids": input_ids,
#       "token_type_ids": token_type_ids,
#       "attention_mask": attention_masks,
#   }, label
# def encode_examples(X, y):
#     # prepare list, so that we can build up final TensorFlow dataset from slices.
#     input_ids_list = []
#     token_type_ids_list = []
#     attention_mask_list = []
#     label_list = []
#     for s in X:
#         bert_input = tokenizer.encode_plus(text=s,add_special_tokens = True, max_length = max_review_length, pad_to_max_length = True, return_attention_mask = True, truncation=True)
#         # print (bert_input)
#         input_ids_list.append(bert_input['input_ids'])
#         token_type_ids_list.append(bert_input['token_type_ids'])
#         attention_mask_list.append(bert_input['attention_mask'])
#     for l in y:
#         label_list.append([l])
#     return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

# def list_splitter(list_to_split, ratio):
#     first_half = int(len(list_to_split) * ratio)
#     return list_to_split[:first_half], list_to_split[first_half:]
# def Convert(tup, di): 
#     for a, b in tup: 
#         di.setdefault(a, b)
#     return di

# # y_pred = numpy.array([4.8483334e-06, 1.4581978e-03, 2.2443533e-03, 1.3272554e-02, 1.3884255e-01,7.5374603e-01])
# # print (str(numpy.argmax(y_pred)))

# # fix random seed for reproducibility
# numpy.random.seed(7)
# tf.random.set_seed(7)
# # load the dataset but only keep the top n words, zero the rest
# top_words = 20000
# # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# # print(len(X_train[0]))
# # print(len(X_train[578]))
# # print(y_train.shape)
# # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# # print(X_train)
# # print(y_train)
# # print(X_train.shape)
# # print(y_train.shape)

# X_train = []
# y_train = []
# with open('data/train.json', 'r') as train_file:
#     for row in train_file:
#         data = json.loads(row)
#         r = float(data['overall'])
#         if "reviewText" in data:
#             post_list = data['reviewText']
#         if "summary" in data:
#             post_list += data['summary']
#         if post_list:    
#             X_train.append(post_list)
#             y_train.append(r)

# X_train = numpy.array(X_train)
# y_train = numpy.array(y_train)
# # X_cmp = X_train
# # y_cmp = y_train
# # idx = numpy.random.permutation(len(X_train))
# # X_train,y_train = X_train[idx], y_train[idx]
# # print ("Xtrain is", X_train[100])
# # print ("ytrain is", y_train[100])
# # print ("idx is", idx[100])
# # print ("Xcmp is", X_cmp[idx[100]])
# # print ("ycmp is", y_cmp[idx[100]])
# # (_, X_test) = list_splitter(X_train, 0.8)
# # (_, y_test) = list_splitter(y_train, 0.8)
# (X_train, X_test) = list_splitter(X_train, 0.8)
# (y_train, y_test) = list_splitter(y_train, 0.8)
# batch_size = 64
# # train dataset
# ds_train_encoded = encode_examples(X_train, y_train).batch(batch_size)
# # test dataset
# ds_test_encoded = encode_examples(X_test, y_test).batch(batch_size)
# # model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
# # loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# # simple early stopping, optional
# # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
# model.fit(ds_train_encoded, epochs=10, validation_data = ds_test_encoded)
# # Final evaluation of the model
# scores = model.evaluate(ds_test_encoded, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# X_pred = []
# with open('data/train.json', 'r') as train_file:
#     for row in train_file:
#         data = json.loads(row)
#         if "reviewText" in data:
#             post_list = data['reviewText']
#         if "summary" in data:
#             post_list += data['summary']
#         if post_list:    
#             X_pred.append(post_list)

# X_pred = numpy.array(X_pred)
# ds_X_pred = encode_examples(X_pred)
# print ("x_pred shape", ds_X_pred.shape)
# y_pred = model.predict(ds_X_pred)
# i = 0
# predictions = open('submission.csv', 'w')
# for l in open('data/rating_pairs.csv'):
#     if l.startswith('userID'):
#         predictions.write(l)
#         continue
#     u,p = l.strip().split('-')
#     # print ("y_pred", y_pred[i], " and ", i)
#     predictions.write(u + '-' + p + ',' + str(numpy.argmax(y_pred[i])) + '.0\n')
#     i += 1
# predictions = open('rating_predictions_bak.csv', 'w')
# for j in y_pred:
#     predictions.write(str(numpy.argmax(j)) + '\n')