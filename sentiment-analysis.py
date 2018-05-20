
# coding: utf-8

# In[1]:


#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train_file = "train.tsv"
test_file = "test.tsv"

TRAIN_MODEL = True
MODEL_NAME = "trained_model_lstm.hdf5"
FILLER_WORD = "unk"

def load_data(file, direc="", sep="\t", header=True):
    csv_path = os.path.join(direc, file)
    if header:
        return pd.read_csv(csv_path, sep=sep, index_col=False)
    else:
        return pd.read_csv(csv_path, sep=sep, index_col=False, header=None)


# In[2]:


train_data = load_data(train_file)
test_data = load_data(test_file)


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


from sklearn .model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(train_data, train_data["Sentiment"]):
    strat_train_set = train_data.loc[train_index]
    strat_val_set = train_data.loc[val_index]
    
train_data = strat_train_set
val_data = strat_val_set


# In[6]:


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
        
   return new_data

train_features = nlp_clean(train_data["Phrase"])
val_features = nlp_clean(val_data["Phrase"])
test_features = nlp_clean(test_data["Phrase"])


# In[7]:


features = []

for indx in range(len(train_features)):
    if len(train_features[indx]) == 0:
        train_feature = train_features[indx]
        train_feature.append(FILLER_WORD)
        train_features[indx] = train_feature
    
    features.append(train_features[indx])
    
for indx in range(len(val_features)):
    if len(val_features[indx]) == 0:
        val_feature = val_features[indx]
        val_feature.append(FILLER_WORD)
        val_features[indx] = val_feature
    
    features.append(val_features[indx])
        
for indx in range(len(test_features)):
    if len(test_features[indx]) == 0:
        test_feature = test_features[indx]
        test_feature.append(FILLER_WORD)
        test_features[indx] = test_feature
    
    features.append(test_features[indx])


# In[8]:


print('\nTraining word2vec...')
word_model = gensim.models.Word2Vec(features, size=300, min_count=1, window=3, iter=100)
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Word embedding shape:', pretrained_weights.shape)


# In[9]:


def word2idx(word):
    return word_model.wv.vocab[word].index

def idx2word(idx):
    return word_model.wv.index2word[idx]


# In[10]:


def encode_sentences(X):
    X_ = []
    for n_indx in range(len(X)):
        ws_ = []
        ws = X[n_indx]
        for w_indx in range(len(ws)):
            wrd = ws[w_indx]
            ws_.append(word2idx(wrd))
            
        X_.append(ws_)
        
    return X_

train_features = encode_sentences(train_features)
val_features = encode_sentences(val_features)
test_features = encode_sentences(test_features)


# In[11]:


y_train = train_data["Sentiment"].values
y_train = np.divide(y_train, 4.0)

y_val = val_data["Sentiment"].values
y_val = np.divide(y_val, 4.0)


# In[12]:


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.models import load_model

MAX_LEN = 100
batch_size = 32
print("Padding the data for LSTM...")
print("Padding train data...")
x_train = sequence.pad_sequences(train_features, maxlen=MAX_LEN)
print("Padding val data...")
x_val = sequence.pad_sequences(val_features, maxlen=MAX_LEN)
print("Padding test data...")
x_test = sequence.pad_sequences(test_features, maxlen=MAX_LEN)


# In[13]:


#x_train = x_train[0:10,]
#y_train = y_train[0:10,]
#x_val = x_val[0:10,]
#y_val = y_val[0:10,]


# In[16]:


droprate = 0.7

try:
    model = load_model(MODEL_NAME)
    print("Loaded saved model: " + MODEL_NAME)
except:
    print("Creating new model: " + MODEL_NAME)
    model = None

if model is None:
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
    model.add(Bidirectional(LSTM(units=emdedding_size, dropout=droprate, recurrent_dropout=droprate)))
    
    '''
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(units=256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(units=32, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    '''

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    
def do_on_epoch_end(epoch, _):
    if(epoch % 2 == 0):
        pred = model.predict(x_val)
        actual = y_val.copy()
        pred = np.round(pred * 4.0).flatten()
        actual = np.round(actual * 4.0).flatten()
        acc = sum(pred == actual)
        print("Accuracy obtained after epoch: " + str(acc) + " / " + str(len(actual)))
    

if TRAIN_MODEL:
    print('\nTraining LSTM model...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              verbose=0,
              validation_data=[x_val, y_val],
              callbacks = [ModelCheckpoint(MODEL_NAME, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1),
                          LambdaCallback(on_epoch_end=do_on_epoch_end)]
             )
    
print('\nValidation LSTM model...')
saved_model = load_model(MODEL_NAME)
score = saved_model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

