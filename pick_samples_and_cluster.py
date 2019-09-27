#!/usr/bin/env python
# coding: utf-8


##  import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)`

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.merge import add


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def load_data(newsfile):
    """
    loads and  cleans the data
    loads the data, and uses only the top 10 classes
    Combining THE WORLDPOST and WORLDPOST categories
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    news_df = pd.read_json(newsfile)
    news_df['category'] = news_df['category'].apply(lambda x: x.replace('THE WORLDPOST','WORLDPOST'))   
    news_categories = news_df['category'].value_counts()
    news_df = news_df[news_df['category'].isin(news_categories[:10].keys())]# category to id

    categories = news_df.groupby('category').size().index.tolist()
    category_int = {}
    int_category = {}
    for i, k in enumerate(categories):
        category_int.update({k:i})
        int_category.update({i:k})

    news_df['c2id'] = news_df['category'].apply(lambda x: category_int[x])
    return news_df, int_category

def get_smaller_sample(df, sample_size = 0.2):
    """
    get s smaller sample of the data
    """
    from sklearn.model_selection import train_test_split
    big_news_df, small_news_df = train_test_split(df, test_size = sample_size, random_state = 42)
    return small_news_df


def tokenize_text(df):
    """
    Tokenizes the text in the news articlces and returns the tokenized 
    words, indices
    Removes articles with 
    """
    from keras.preprocessing.text import Tokenizer, text_to_word_sequence
    tokenizer = Tokenizer()
    # using headlines and short_description as input X
    df.loc[:,'text'] = df.loc[:,'headline'] + " " + df.loc[:,'short_description']
    tokenizer.fit_on_texts(df.loc[:,'text'])
    df.loc[:,'word_ids'] = tokenizer.texts_to_sequences(df.loc[:,'text'])
    df.loc[:,'word_length'] = df.loc[:,'word_ids'].apply(lambda i: len(i)).values
    word_index = tokenizer.word_index
    # created reverse dictionary withvalue, key pair
    word_index_vk = dict( (v,k) for k,v in word_index.items())
    df.loc[:,'words'] = df.loc[:,'word_ids'].apply(lambda x: [word_index_vk[i] for i in x])
    # delete some empty and short data
    df = df[df['word_length']>5]
    return df, word_index

def vectorize_word(sent, model):
    """
    vectorizes the text for each document in the list using 
    gensim's word2vect model
    returns a numpy vector
    """    
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model.wv[w]
            else:
                sent_vec = np.add(sent_vec, model.wv[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw

def vectorize_words(df, model):
    """
    vectorizes all the docucments using the Word2Vec model
    reurns a WordVector of 100 vector for each 
    document in the collection
    """
    from gensim.models import Word2Vec
    from nltk.cluster import KMeansClusterer
    import nltk
    import numpy as np 
    X=[]
    for sentence in df['words']:
        X.append(vectorize_word(sentence, model))
    nrows = len(X)
    X = [j for i in np.array(X) for j in i]
    X = np.array(X).reshape((nrows, model.vector_size))
    return X


def cluster_documents(df, model, n_clusters = 10):
    """
    cluster the documents based on number of clusters, 
    given the document df
    
    vectorizes the words in the documents
    
    k_mean_cluster for given  number of clusters
    
    returns the label and clusters
    """
    X = vectorize_words(df, model)
    from sklearn import cluster
    from sklearn import metrics
    kmeans = cluster.KMeans(n_clusters)
    kmeans.fit(X)
    df.loc[:,'cluster_labels'] = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return df, X, centroids, kmeans.labels_

def calculate_nearest_n(df, model, n_clusters = 10,  sample_ratio=.2):
                        
    """
    get the nearest n-items from the  centroid
    """
    from collections import Counter
    from sklearn.metrics.pairwise import euclidean_distances
    df, X, centroids, labels = cluster_documents(df, model, n_clusters)
    import numpy as np
    sample_ids = np.array([],int)
    labels_sample = []
    for i in range(n_clusters):
        cluster_ids = np.where(labels == i)
        X1 = X[cluster_ids]
        cluster_size = len(cluster_ids[0])
        cluster_sample_size  = int(sample_ratio*cluster_size)
        ids = np.argsort(euclidean_distances(X1, centroids[i].reshape(1,-1))
                         .transpose()).flatten()[:cluster_sample_size]
        sample_ids = np.append(sample_ids,cluster_ids[0][ids], axis = 0)
    return df, sample_ids


def plot_knn_for_k(df, model, kstart = 2, kend = 42, kinc  = 2):
    """
    plot the wss for different values of k 
    to specify the  right number of clusters
    """
    from sklearn import cluster
    from sklearn import metrics
    from matplotlib import pyplot as plt
    plt.rcParams['figure.figsize']=(15,10)
    plt.rcParams.update({'font.size': 22})
    X = vectorize_words(df, model)
    kscore = []
    for i in range(kstart,kend,kinc):
        kmeans = cluster.KMeans(n_clusters=i,  n_jobs = 4)
        kmeans.fit(X)
        kscore.append(kmeans.inertia_)
    plt.plot(list(range(kstart,kend,kinc)),kscore,'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Weighted sum of the squared distances')
    return kscore

def plot_tsne(df, model, n_components = 2):
    """
    plot ths stochastic t-neighbor embedding for 
    visualize the word vector
    """
    from sklearn.manifold import TSNE
    import matplotlib
    X = vectorize_words(df,model)

    plt.rcParams['figure.figsize']=(15,10)
    doc_proj = TSNE(n_components=2, random_state=42, ).fit_transform(X)
   
    sc = plt.scatter(doc_proj[:,0], doc_proj[:,1],c=df['cluster_labels'], s = 5,
                     cmap = plt.cm.get_cmap('RdYlBu'))
    plt.colorbar(sc)
    plt.rcParams.update({'font.size': 22})
    return doc_proj


def pivot_category_and_cluster_label(df):
    """
    pivots the category cand cluster_label
    for gettting distribution of clusters 
    in each category
    """
    plt.rcParams['figure.figsize']=(15,10)
    cat_clust_label_distr = df.groupby(['category','cluster_labels'])['cluster_labels'].count().unstack().transpose()
    cat_clust_label_distr.plot(kind='bar', stacked = True)
    plt.rcParams.update({'font.size': 16})
    return cat_clust_label_distr


class Attention(Layer):
    """
    Attention class for building an attention model
    """
    from keras.engine.topology import Layer
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        print(self.name)
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                shape = (input_shape[-1],),
                                initializer=self.init,
                                regularizer=self.W_regularizer,
                                constraint=self.W_constraint),
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                    shape = (input_shape[1],),
                                    initializer='zero',
                                    regularizer=self.b_regularizer, 
                                    constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    

def get_embedding_layer(df, word_index, embedding_file_path, EMBEDDING_DIM, maxlen):
    """
    returns  the embedding_layer and word index, given the glove file
    
    """
    import numpy as np
    
    embeddings_index = {}
    f = open(embedding_file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    #df, word_index = tokenize_text(df)
    print('Found %s unique tokens.' % len(word_index))
    print('Total %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)
    return embedding_layer, word_index


def build_attention_model(df, word_index, embedding_file_path, EMBEDDING_DIM, maxlen):
    """
    builds an attention model
    """
    embedding_layer, word_index = get_embedding_layer(df, word_index, 
                                        embedding_file_path, EMBEDDING_DIM, maxlen)
    lstm_layer = LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)
    inp = Input(shape=(maxlen,), dtype='int32')
    embedding= embedding_layer(inp)
    x = lstm_layer(embedding)
    x = Dropout(0.25)(x)
    merged = Attention(maxlen)(x)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.25)(merged)
    merged = BatchNormalization()(merged)
    outp = Dense(len(int_category), activation='softmax')(merged)

    AttentionLSTM = Model(inputs=inp, outputs=outp)
    AttentionLSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    AttentionLSTM.summary()
    return AttentionLSTM

def plot_accuracy(attlstm_history):
    """
    plots the accuracy
    """
    acc = attlstm_history.history['acc']
    val_acc = attlstm_history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.rcParams['figure.figsize']=(15,10)
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'ro-', label='Training acc')
    plt.plot(epochs, val_acc, 'bo-', label='Validation acc')
    plt.rcParams.update({'font.size': 16})
    plt.legend()

def plot_loss(attlstm_history):
    """
    plots the losses
    """
    loss = attlstm_history.history['loss']
    val_loss = attlstm_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.rcParams['figure.figsize']=(15,10)
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'ro-', label='Training loss')
    plt.plot(epochs, val_loss, 'bo-', label='Validation loss')
    plt.rcParams.update({'font.size': 16})
    plt.legend()

def generate_train_test_set(df):
    """
    generates the training and test samples
    """
    
    # using 50 for padding length
    maxlen = 50
    df, word_index = tokenize_text(df)
    X = list(sequence.pad_sequences(df.word_ids, maxlen=maxlen))
    # prepared data 

    X = np.array(X)
    Y = np_utils.to_categorical(list(df.c2id))
    # and split to training set and validation set
    seed = 29
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
    return word_index, x_train, x_val, y_train, y_val


def run_lstm(df, embedding_file_path, EMBEDDING_DIM, maxlen, epochs = 20):  
    """
    runs the lstm model
    """
    word_index, x_train, x_val, y_train, y_val = generate_train_test_set(df)
    AttentionLSTM =  build_attention_model(df, word_index, embedding_file_path, EMBEDDING_DIM, maxlen)
    attlstm_history = AttentionLSTM.fit(x_train, 
                                    y_train, 
                                    batch_size=128, 
                                    epochs=epochs, 
                                    validation_data=(x_val, y_val))
    predictions = AttentionLSTM.predict(x_val)
    return attlstm_history, predictions


def  main():
    from gensim.models import Word2Vec
    EMBEDDING_DIM = 100
    maxlen = 50
    embedding_file_path = 'data/glove.6B.100d.txt'
    small_news_df, int_category =  load_data('data/Small_News_Category_Dataset_v2.json')
    #small_news_df = get_smaller_sample(news_df,sample_size = 0.2)
    small_news_df, word_index = tokenize_text(small_news_df)
    model = Word2Vec(small_news_df['words'], min_count=1)
    #small_news_df, X, centroids, labels = cluster_documents(small_news_df, nclusters = 10)
    small_news_df, sample_ids  = calculate_nearest_n(small_news_df, model, n_clusters = 10, sample_ratio = 0.2)
    samples_for_labeling_df = small_news_df.iloc[sample_ids]
    doc_proj = plot_tsne(samples_for_labeling_df, model)

    ## Model for sampled using clustereing
    att_lstm_history_full, predictions_full = run_lstm(small_news_df, embedding_file_path, 
                                                       EMBEDDING_DIM, maxlen)

    
    # Model randomly sampled
    att_lstm_history_sampled, predictions_sampled = run_lstm(samples_for_labeling_df, 
                                                             embedding_file_path, EMBEDDING_DIM, maxlen)
    
    random_sample_df = get_smaller_sample(small_news_df, sample_size = 0.2)
    att_lstm_history_random_sample, predictions_random = run_lstm(random_sample_df,embedding_file_path, 
                                                                  EMBEDDING_DIM, maxlen)

    

    
    plot_accuracy(att_lstm_history_sampled)
    plot_loss(att_lstm_history_sampled)

    plot_accuracy(att_lstm_history_random_sample)
    plot_loss(att_lstm_history_random_sample)

    plot_accuracy(att_lstm_history_full)
    plot_loss(att_lstm_history_full)

if __name__ == '__main__':
    main()



