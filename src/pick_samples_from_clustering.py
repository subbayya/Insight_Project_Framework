"""
pick samples for labeling using  clustering
"""


##  import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)`

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6,6)


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
    news_df = pd.read_json(newsfile, lines = True)
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
    big_news_df, small_news_df = train_test_split(df, test_size = sample_size, stratify = df['category'],
                                                  random_state = 42)
    return big_news_df, small_news_df


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
    cat_clust_label_distr = df.groupby(['category','cluster_labels'])                                ['cluster_labels'].count().unstack().transpose()
    cat_clust_label_distr.plot(kind='bar', stacked = True)
    plt.rcParams.update({'font.size': 16})
    plt.legend(frameon=False, labelspacing=1, loc='upper center', fontsize = 12)
    return cat_clust_label_distr




def get_embedding_layer(embedding_file_path, EMBEDDING_DIM, maxlen):
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
   
    print('Total %s word vectors.' % len(embeddings_index))
    word_index = {w: i for i, w in enumerate(embeddings_index.keys(), 1)}
    print('Found %s unique tokens.' % len(word_index))
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

    

def build_lstm_model(df, embedding_layer,maxlen):
    """
    builds an attention model
    """
    #embedding_layer, word_index = get_embedding_layer(df, word_index, 
    #                                    embedding_file_path, EMBEDDING_DIM, maxlen)
    lstm_layer = LSTM(300, dropout=0.25, recurrent_dropout=0.25, return_sequences=False)
    inp = Input(shape=(maxlen,), dtype='int32')
    embedding= embedding_layer(inp)
    x = lstm_layer(embedding)
    x = Dropout(0.25)(x)
    
    merged = Dense(256, activation='relu')(x)
    merged = Dropout(0.25)(merged)
    merged = BatchNormalization()(merged)
    #merged = Flatten(merged)
    nclasses = len(np.unique(df['category']))

    outp = Dense(nclasses, activation='softmax')(merged)

    lstm_model = Model(inputs=inp, outputs=outp)
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    lstm_model.summary()
    return lstm_model


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
    X = list(sequence.pad_sequences(df.word_ids, maxlen=maxlen))
    # prepared data 

    X = np.array(X)
    Y = np_utils.to_categorical(list(df.c2id))
    # and split to training set and validation set
    seed = 29
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, 
                                                      stratify = Y, random_state=seed)
    return x_train, x_val, y_train, y_val


def test_set_for_prediction(df):
    """
    generates the training and test samples
    """
    
    # using 50 for padding length
    maxlen = 50
    #df, word_index = tokenize_text(df)
    X = list(sequence.pad_sequences(df.word_ids, maxlen=maxlen))
    # prepared data 

    X = np.array(X)
    Y = np_utils.to_categorical(list(df.c2id))
    # and split to training set and validation set
    seed = 29
   
    return X, Y

def run_lstm(df, embedding_layer, maxlen, epochs = 10):  
    """
    runs the lstm model
    """
    from sklearn.utils import class_weight
    x_train, x_val, y_train, y_val = generate_train_test_set(df)
    lstm =  build_lstm_model(df, embedding_layer, maxlen)
    weighted_metrics = class_weight.compute_sample_weight('balanced', y_train)

    lstm_history = lstm.fit(x_train, 
                                    y_train, 
                                    batch_size=128, 
                                    epochs=epochs, 
                                    validation_data=(x_val, y_val))
    predictions = lstm.predict(x_val)
    cm = pd.DataFrame(confusion_matrix(y_val.argmax(axis=1), predictions.argmax(axis=1)))
    return lstm, lstm_history, predictions, cm

def pick_samples_with_low_scores(predictions, sample_ratio):
    import numpy as np
    max_scores = np.max(predictions, axis = 1)
    nsamples = int(len(max_scores)*sample_ratio)
    return np.argsort(max_scores)[:nsamples]



def tokenize_text_model(df, word_index):
    """
    Tokenizes the text in the news articlces and returns the tokenized 
    words, indices
    Removes articles with 
    """
    from keras.preprocessing.text import Tokenizer, text_to_word_sequence
    # using headlines and short_description as input X
    df.loc[:,'text'] = df.loc[:,'headline'].astype(str) + " " + df.loc[:,'short_description'].astype(str)
    df.loc[:, 'words'] = df.loc[:,'text'].apply(text_to_word_sequence)
    df.loc[:,'word_ids'] = df.loc[:,'words'].apply(lambda x:                                         [word_index.get(i) for i in x if word_index.get(i)])
    df.loc[:,'word_length'] = df.loc[:,'word_ids'].apply(lambda i: len(i)).values
    df = df[df['word_length']>5]
  

    return df




def get_precision_recall_accuracy(cm):
    """
    get the preicison, recall, accuracy given the confusion matrix
        """
    recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
    precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
    accuracy = np.sum(np.diag(cm))/np.sum(np.sum(cm))
    return accuracy, precision, recall


def  main():
    from gensim.models import Word2Vec
    EMBEDDING_DIM = 100
    maxlen = 50
    small_news_df, int_category =  load_data('input/News_Category_Dataset_v2.json')
    #small_news_df = get_smaller_sample(news_df,sample_size = 0.2)
    small_news_df, word_index = tokenize_text(small_news_df)
    model = Word2Vec(small_news_df['words'], min_count=5, )
    #small_news_df, X, centroids, labels = cluster_documents(small_news_df, nclusters = 10)
    small_news_df, sample_ids  = calculate_nearest_n(small_news_df, model, n_clusters = 10, sample_ratio = 0.2)
    samples_for_labeling_df = small_news_df.iloc[sample_ids]
    doc_proj = plot_tsne(samples_for_labeling_df, model)

    ## Model for sampled using clustereing
    att_lstm_history_full, predictions_full = run_lstm(small_news_df, embedding_file_path,
                                                       EMBEDDING_DIM, maxlen)

    
    # Model randomly sampled
    att_lstm_history_sampled, predictions_sampled = run_lstm(samples_for_labeling_df,
                                                             embedding_file_path, 
                                                             EMBEDDING_DIM, maxlen)
    
    random_sample_df = get_smaller_sample(small_news_df, sample_size = 0.2)
    att_lstm_history_random_sample, predictions_random = run_lstm(random_sample_df,embedding_file_path, 
                                                                 EMBEDDING_DIM, maxlen)
    

if  __name__ == "__main__":
    main()


