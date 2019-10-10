"""
Pick samples labeling for the given dataset
"""
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


## Buiilding the model 


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




def main():
    EMBEDDING_DIM = 100
    maxlen = 50
    embedding_file_path = 'input/glove.6B.100d.txt'
    embedding_layer, word_index = get_embedding_layer(embedding_file_path, EMBEDDING_DIM, maxlen)
    news_df, int_category =  load_data('input/News_Category_Dataset_v2.json')
    news_df = tokenize_text_model(news_df, word_index)
    big_news_df,  small_news_df  = get_smaller_sample(news_df, sample_size = 0.05) 
    big_news_df, big_news_test_df =  get_smaller_sample(big_news_df, sample_size = 0.05) 
    big_news_df.shape, small_news_df.shape, big_news_test_df.shape
    lstm, lstm_history, lstm_predictions, lstm_cm = run_lstm(small_news_df,                                                          
                                                         embedding_layer,  maxlen, epochs = 20)


    # Make predictions for the larger set
    [X, y] = test_set_for_prediction(big_news_df)
    predictions = lstm.predict(X)

    # pick samples for predictions
    ids = pick_samples_with_low_scores(predictions, 0.01)
    sample_news_df = big_news_df.iloc[ids] 
    small_plus_sample_df = pd.concat([small_news_df, sample_news_df], ignore_index=True)

    lstm_selected, lstm_history_selected, lstm_predictions_selected, lstm_cm_selected = run_lstm(small_plus_sample_df,                                                          
                                                         embedding_layer,  maxlen, epochs = 20)

    _, random_sample_news_df =  get_smaller_sample(big_news_df, sample_size = 0.0105)
    small_plus_random_sample_df = pd.concat([small_news_df, random_sample_news_df], ignore_index=True)

    lstm_random, lstm_history_random, lstm_predictions_random, lstm_cm_random = run_lstm(small_plus_random_sample_df,                                                          
                                                         embedding_layer,  maxlen, epochs = 20)

    predictions_selected = lstm_selected.predict(X)
    predictions_random  = lstm_random.predict(X)
    cm_selected =  pd.DataFrame(confusion_matrix(y.argmax(axis=1), predictions_selected.argmax(axis=1)))
    cm_random  =   pd.DataFrame(confusion_matrix(y.argmax(axis=1), predictions_random.argmax(axis=1)))
    print('Confusion_matrix from selected samples')
    print(cm_selected)
    print('Confusion_matrix from random samples')
    print(cm_random)  
    print('Accuracy, Precision, recall from selected samples')
    accuracy_s, precision_s, recall_s = get_precision_recall_accuracy(cm_selected)
    print(accuracy_s, precision_s, recall_s)
    print('Accuracy, Precision, recall from random samples')
    accuracy_r, precision_r, recall_r = get_precision_recall_accuracy(cm_random)
    print(accuracy_r, precision_r, recall_r)

if __name__ == "__main__":
    main()


