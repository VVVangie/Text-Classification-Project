#!/usr/bin/env python
# coding: utf-8

# ### Name: Junren Jiang
# ### Github Username: JunrenRex
# ### USC ID: 4923051887

# In[12]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install tensorflow')


# In[13]:


from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import os
import re
import statistics
from collections import Counter
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Flatten,Dense,Dropout,MaxPooling1D,Conv1D,LSTM
from tensorflow.keras import Input


# # 1. Text Classification
# ### It is highly recommended that you complete this project using Keras and Python.
# ### (a) In this problem, we are trying to build a classifier to analyze the sentiment of reviews. You are provided with text data in two folders: one folder involves positive reviews, and one folder involves negative reviews.
# ## (b) Data Exploration and Pre-processing
# ### i. You can use binary encoding for the sentiments , i.e y = 1 for positive sentiments and y =-1 for negative sentiments.
# ### ii. The data are pretty clean. Remove the punctuation and numbers from the data.

# In[14]:


path1 = '../data/neg'
neg_file = os.listdir(path1)
neg_file.sort()
neg_set = []
for f1 in neg_file:
    location1 = os.path.join(path1, f1)
    with open(location1, 'r', encoding='utf-8') as file1:
        neg_con = file1.read()
        neg_con = neg_con.replace('\n','')
        neg_content = re.sub(r'[^a-zA-Z\s]',' ',neg_con)
        neg_set.append(neg_content)
        file1.close()
neg_frame = pd.DataFrame(neg_set)
neg_frame.columns = ['review']
# Use label y = 0 for negative sentiments
neg_frame['sentiment'] = 0
neg_frame


# In[15]:


path2 = '../data/pos'
pos_file = os.listdir(path2)
pos_file.sort()
pos_set = []
for f2 in pos_file:
    location2 = os.path.join(path2,f2)
    with open(location2,'r',encoding='utf-8')as file2:
        pos_con = file2.read()
        pos_con = pos_con.replace('\n','')
        pos_content = re.sub(r'[^a-zA-Z\s]',' ',pos_con)
        pos_set.append(pos_content)
        file2.close()
pos_frame = pd.DataFrame(pos_set)
pos_frame.columns = ['review']
# Use label y = 1 for positive sentiments
pos_frame['sentiment'] = 1
pos_frame


# ### b.iii. The name of each text file starts with cv_number. Use text files 0-699 in each class for training and 700-999 for testing.

# In[16]:


neg_train = neg_frame.iloc[:700,:]
neg_test = neg_frame.iloc[700:,:]
pos_train = pos_frame.iloc[:700,:]
pos_test = pos_frame.iloc[700:,:]
training_set = pd.concat([neg_train,pos_train])
test_set = pd.concat([neg_test,pos_test])
test_set


# ### b.iv. Count the number of unique words in the whole dataset (train + test) and print it out.

# In[17]:


full_set = pd.concat([neg_frame,pos_frame])
unique_word = set(full_set['review'].str.lower().str.findall("\w+").sum())
print('The number of unique words is:', len(unique_word))
unique_word


# ### b.v. Calculate the average review length and the standard deviation of review lengths. Report the results.

# In[18]:


review_length = []
for c in full_set['review']:
    review_len = len(c.split(' '))
    review_length.append(review_len)
print('The average review length is: ', int(statistics.mean(review_length)))
print('The standard deviation of review lengths is: ', statistics.stdev(review_length))


# ### b.vi. Plot the histogram of review lengths.

# In[19]:


hist_data = Counter(review_length)
print(hist_data)


# In[20]:


plt.hist(hist_data)
plt.xlabel('Review Lengths')
plt.ylabel('Frequency')


# ### b.vii. To represent each text (= data point), there are many ways. In NLP/Deep Learning terminology, this task is called tokenization. It is common to represent text using popularity/ rank of words in text. The most common word in the text will be represented as 1, the second most common word will be represented as 2, etc. Tokenize each text document using this method.

# In[21]:


train_content = training_set['review'].tolist()
t1 = Tokenizer()
t1.fit_on_texts(train_content)
sequence1 = t1.texts_to_sequences(train_content)
#print("The sequences generated from text in training set are : ",sequence1)
sequence1


# In[22]:


test_content = test_set['review'].values
#t2 = Tokenizer()
#t2.fit_on_texts(test_content)
sequence2 = t1.texts_to_sequences(test_content)
#print("The sequences generated from text in test set are : ",sequence2)
sequence2


# ### b.viii. Select a review length L that 70% of the reviews have a length below it. If you feel more adventurous, set the threshold to 90%.

# In[23]:


print(len(review_length)*0.7)


# In[24]:


review_length.sort()
print('The selected review lenght L is: ',review_length[1400])


# ### b.ix. Truncate reviews longer than L words and zero-pad reviews shorter than L so that all texts (= data points) are of length L.

# In[26]:


sequence1_train = pad_sequences(sequence1, padding='post',truncating='post',maxlen=1004,value=0)
sequence1_train


# In[28]:


sequence2_test = pad_sequences(sequence2, padding='post',truncating='post',maxlen=1004,value=0)
sequence2_test


# ## (c) Word Embeddings
# ### i. One can use tokenized text as inputs to a deep neural network. However, a recent breakthrough in NLP suggests that more sophisticated representations of text yield better results. These sophisticated representations are called word embeddings. “Word embedding is a term used for representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.” . Most deep learning modules (including Keras) provide a convenient way to convert positive integer representations of words into a word embedding by an “Embedding layer.” The layer accepts arguments that define the mapping of words into embeddings, including the maximum number of expected words also called the vocabulary size (e.g. the largest integer value). The layer also allows you to specify the dimension for each word vector, called the “output dimension.” We would like to use a word embedding layer for this project. Assume that we are interested in the top 5,000 words. This means that in each integer sequence that represents each document, we set to zero those integers that represent words that are not among the top 5,000 words in the document. If you feel more adventurous, use all the words that appear in this corpus. Choose the length of the embedding vector for each word to be 32. Hence, each document is represented as a 32 × L matrix.

# In[29]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_content)
train_seq = tokenizer.texts_to_sequences(train_content)
test_seq = tokenizer.texts_to_sequences(test_content)
train_seq = pad_sequences(train_seq, padding='post', maxlen=1004,value=0)
test_seq = pad_sequences(test_seq, padding='post', maxlen=1004,value=0)
train_seq


# In[30]:


model= Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=1004))#input_dim=vocab, output_dim=emb_dim,input_length=Length of input sequences
model.compile()
embedding_train = model.predict(train_seq)
print(model.output_shape)
print(embedding_train)


# In[31]:


embedding_test = model.predict(test_seq)
print(model.output_shape)
print(embedding_test)


# ### c.ii. Flatten the matrix of each document to a vector.

# In[32]:


model.add(Flatten())
model.compile()
flatten_train = model.predict(train_seq)
print(model.output_shape)
print(len(flatten_train))
print(flatten_train)


# In[33]:


flatten_test = model.predict(test_seq)
print(model.output_shape)
print(flatten_train)


# ## (d) Multi-Layer Perceptron
# ### i. Train a MLP with three (dense) hidden layers each of which has 50 ReLUs and one output layer with a single sigmoid neuron. Use a dropout rate of 20% for the first layer and 50% for the other layers. Use ADAM optimizer and binary cross entropy loss (which is equivalent to having a softmax in the output). To avoid overfitting, just set the number of epochs as 2. Use a batch size of 10.

# In[34]:


train_label = np.array(training_set['sentiment'])
test_label = np.array(test_set['sentiment'])
X_train = np.array(train_seq)
X_test = np.array(test_seq)


# In[35]:


mlp_model= Sequential()
mlp_model.add(Embedding(input_dim=5000, output_dim=32, input_length=1004))
mlp_model.add(Flatten())
mlp_model.add(Input(shape=(50,)))
mlp_model.add(Dense(50, activation='relu'))
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(50, activation='relu'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(50, activation='relu'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(1, activation='sigmoid'))
mlp_model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['BinaryAccuracy'])
mlp_model.fit(X_train,train_label,epochs=2,batch_size=10)


# In[36]:


mlp_model.summary()


# ### d.ii. Report the train and test accuracies of this model.

# In[37]:


mlp_train_loss,mlp_train_accuracy = mlp_model.evaluate(X_train,train_label)
print('The train accuracy is: ', mlp_train_accuracy)


# In[38]:


mlp_test_loss,mlp_test_accuracy = mlp_model.evaluate(X_test,test_label)
print('The test accuracy is: ', mlp_test_accuracy)


# ## (e) One-Dimensional Convolutional Neural Network:
# ### Although CNNs are mainly used for image data, they can also be applied to text data, as text also has adjacency information. Keras supports one-dimensional convolutions and pooling by the Conv1D and MaxPooling1D classes respectively.
# ### i. After the embedding layer, insert a Conv1D layer. This convolutional layer has 32 feature maps , and each of the 32 kernels has size 3, i.e. reads embedded word representations 3 vector elements of the word embedding at a time. The convolutional layer is followed by a 1D max pooling layer with a length and stride of 2 that halves the size of the feature maps from the convolutional layer. The rest of the network is the same as the neural network above.

# In[39]:


train_label = train_label.reshape((-1,1))
test_label = test_label.reshape((-1,1))


# In[40]:


cnn_model= Sequential()
cnn_model.add(Embedding(input_dim=5000, output_dim=32, input_length=1004))
#cnn_model.add(Flatten())
cnn_model.add(Conv1D(32, 3, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2, strides=None))
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['BinaryAccuracy'])
cnn_model.fit(X_train,train_label,epochs=2,batch_size=10)


# In[41]:


cnn_model.summary()


# ### e.ii. Report the train and test accuracies of this model.

# In[42]:


cnn_train_loss,cnn_train_accuracy = cnn_model.evaluate(X_train,train_label)
print('The train accuracy is: ', cnn_train_accuracy)


# In[43]:


cnn_test_loss,cnn_test_accuracy = cnn_model.evaluate(X_test,test_label)
print('The test accuracy is: ', cnn_test_accuracy)


# ## (f) Long Short-Term Memory Recurrent Neural Network:
# ### The structure of the LSTM we are going to use is shown in the following figure.
# ### i. Each word is represented to LSTM as a vector of 32 elements and the LSTM is followed by a dense layer of 256 ReLUs. Use a dropout rate of 0.2 for both LSTM and the dense layer. Train the model using 10-50 epochs and batch size of 10.

# In[44]:


lstm_model= Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=32, input_length=1004))
#lstm_model.add(Flatten())
lstm_model.add(LSTM(32))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(256, activation='relu'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['BinaryAccuracy'])
lstm_model.fit(X_train,train_label,epochs=30,batch_size=10)


# ### f.ii. Report the train and test accuracies of this model.

# In[45]:


lstm_train_loss,lstm_train_accuracy = lstm_model.evaluate(X_train,train_label)
print('The train accuracy is: ', lstm_train_accuracy)


# In[46]:


lstm_test_loss,lstm_test_accuracy = lstm_model.evaluate(X_test,test_label)
print('The test accuracy is: ', lstm_test_accuracy)


# #### References:
# #### https://towardsdatascience.com/a-guide-to-text-classification-and-sentiment-analysis-2ab021796317
# #### https://blog.csdn.net/fengdu78/article/details/121463586
# #### https://blog.csdn.net/weixin_63010499/article/details/122544489
# #### https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
# #### https://machinelearningknowledge.ai/keras-tokenizer-tutorial-with-examples-for-fit_on_texts-texts_to_sequences-texts_to_matrix-sequences_to_matrix/
# #### https://www.danli.org/2021/06/21/hands-on-machine-learning-keras/
# #### https://blog.csdn.net/AlanxZhang/article/details/120662453
