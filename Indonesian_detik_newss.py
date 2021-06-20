# -*- coding: utf-8 -*-
"""
# <center> News Title NLP Classification

---

<center> [dataset](https://www.kaggle.com/ibamibrahim/indonesian-news-title)

<small> *note: the output was run on CPU mode*
"""

import re
import nltk
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from multiprocessing.dummy import Pool as ThreadPool
from pip._internal import main as pipmain

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

pipmain(['install', 'sastrawi'])
pipmain(['install', 'tensorflow-addons'])
# from nltk.tokenize import word_tokenize
tf.random.set_seed(233)
np.random.seed(233)

train = pd.read_csv('indonesian-news-title.csv')

train = train.drop_duplicates(subset=['title'], keep=False).reset_index(drop=True)
train = train.loc[:, ['title', 'category']]

"""# Cleaning Dataset"""

# Removing the stopwords from text


def remove_stopwords(text):
    final_text = []
    factory_stop = StopWordRemoverFactory()
    stop_sastrawi = factory_stop.get_stop_words()
    stop_corpus_in = stopwords.words('indonesian')
    stop_corpus_en = stopwords.words('english')
    #     more_stopword = ['dengan', 'ia', 'bahwa','oleh', 'yg', 'dlm']
    # more_stopword = ['dengan', 'ia', 'bahwa','oleh', 'yg', 'dlm', 'deh',
    #                  'sih', 'lg', 'krn', 'tlg', 'jk', 'sdh', 'tp', 'dpt',
    #                  'gt', '&', 'nya', 'duh', 'dih', 'kok', 'nah', 'an',
    #                 'donk', 'dung', 'dong', 'duns']

    stop = stop_sastrawi + stop_corpus_in + stop_corpus_en
    stop = set(stop)

    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return ' '.join(final_text)

# Removing all the noisy text


def denoise_text(text):
    '''full process'''
    text = remove_stopwords(text)  # can't use with threadpool
    text = re.sub('\d', '', text).lower()
    text = ' '.join([i.strip() for i in text.split()])
    return text


train['title_clean'] = train.title.apply(denoise_text)
train.head()

"""# Data Preparation"""

# # for class weight 'ON'
# from sklearn.model_selection import train_test_split

# title_text = train.title_clean.values
# cat_text = train.category.values

# train_x, valid_x, train_y, valid_y = train_test_split(title_text, cat_text, stratify=cat_text, test_size=.2)

# train_x.shape, valid_x.shape, train_y.shape, valid_y.shape

# split first before oversampling to prevent data overfit!"""

train_data, valid_data = train_test_split(
    train, stratify=train['category'], test_size=.2)

train_data.shape, valid_data.shape

# oversampling (set class_weight to None)
nws = train[train.category == 'news']
hot = train[train.category == 'hot']
fnc = train[train.category == 'finance']
trv = train[train.category == 'travel']
net = train[train.category == 'inet']
hlt = train[train.category == 'health']
oto = train[train.category == 'oto']
fdd = train[train.category == 'food']
spr = train[train.category == 'sport']

hot_ov = pd.concat([hot, hot])
fnc_ov = pd.concat([fnc, fnc])
trv_ov = pd.concat([trv, trv, trv, trv, trv])
net_ov = pd.concat([net, net, net, net, net])
hlt_ov = pd.concat([hlt, hlt, hlt, hlt, hlt, hlt])
oto_ov = pd.concat([oto, oto, oto, oto, oto, oto, oto])
fdd_ov = pd.concat([fdd, fdd, fdd, fdd, fdd, fdd, fdd])
spr_ov = pd.concat([spr, spr, spr, spr, spr, spr,
                    spr, spr, spr, spr, spr, spr, spr])

x_over = pd.concat([nws, hot_ov, fnc_ov, trv_ov, net_ov,
                    hlt_ov, oto_ov, fdd_ov, spr_ov])
tr_over = x_over.sample(frac=1).reset_index(drop=True)

train_x = tr_over.title_clean.values
train_y = tr_over.category.values
valid_x = valid_data.title_clean.values
valid_y = valid_data.category.values

print(Counter(train_y))
print(Counter(valid_y))

# Normalize
le = LabelEncoder()

y_tr = le.fit_transform(train_y)
y_val = le.transform(valid_y)

# Tokenize
VAL_PDTR = 'post'

tokenizer = Tokenizer(num_words=5500, oov_token='<OOV>')
tokenizer.fit_on_texts(train_x)

sekuens_train = tokenizer.texts_to_sequences(train_x)
sekuens_valid = tokenizer.texts_to_sequences(valid_x)

padded_train = pad_sequences(
    sekuens_train, truncating=VAL_PDTR, padding=VAL_PDTR, maxlen=12)
padded_valid = pad_sequences(
    sekuens_valid, truncating=VAL_PDTR, padding=VAL_PDTR, maxlen=12)

"""# Modelling"""

SCHEDULE = tf.optimizers.schedules.PiecewiseConstantDecay(
    [1407*20, 1407*30], [1e-3, 1e-4, 1e-5])
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
LR = 1e-1 * schedule(step)


def WD(): return 1e-4 * SCHEDULE(step)


OPTIMIZER = AdamW(learning_rate=SCHEDULE, weight_decay=WD)

# imbalanced
CLW = {
    5: 1,
    3: 2,
    0: 2,
    8: 5,
    4: 5,
    2: 6,
    6: 7,
    1: 7,
    7: 13
}


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=5500, output_dim=32))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(9, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=OPTIMIZER, metrics=['accuracy'])
    return model


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_metrics(history, metrics):
    '''
    Plotting metrik from training process
    '''
    plt.title("Model Performance")
    for metric in metrics:
        plt.plot(history.history[metric], label=metric)
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


cb = tf.keras.callbacks

model = build_model()
stopper = cb.EarlyStopping(patience=3, min_delta=0.05, baseline=0.8,
                          mode='min', monitor='val_loss', restore_best_weights=True,
                          verbose=1)

tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')

total_t0 = time.time()

hist = model.fit(padded_train, y_tr, epochs=10, validation_data=(
    padded_valid, y_val), callbacks=[stopper], class_weight=None)

print('')
print('Training complete!')

print('Total training took {:} (h:mm:ss)'.format(
    format_time(time.time()-total_t0)))

"""# Evaluation"""

# fig, axs = plt.subplots(1, 2)

# axs[1].plot(plot_metrics(hist, ['accuracy', 'val_accuracy']))
# axs[0].plot(plot_metrics(hist, ['loss', 'val_loss']))

eval_df = pd.DataFrame(hist.history)
length = len(eval_df)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

eval_df[['loss', 'val_loss']].plot(ax=ax[0])
ax[0].set(title='Loss', xlabel='Epoch(s)', xticks=range(0, length, 2))

eval_df[['accuracy', 'val_accuracy']].plot(ax=ax[1])
ax[1].set(title='Accuracy', xlabel='Epoch(s)', xticks=range(0, length, 2))

text_sample = ['L. Hamilton kembali menjuarai kompetisi F1 untuk yang ke sekian kalinya',
              'sayang keuangan di perusahaan tersebut kini tengah naik turun']
text_sm_cl = [denoise_text(i) for i in text_sample]

sekuens_sample = tokenizer.texts_to_sequences(text_sm_cl)
padded_sample = pad_sequences(
    sekuens_sample, truncating=VAL_PDTR, padding=VAL_PDTR, maxlen=12)

le.inverse_transform(np.argmax(model.predict(padded_sample), axis=1))

"""**<center> Thank You** 
<br>&copy;2021
"""
