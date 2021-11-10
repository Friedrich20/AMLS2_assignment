#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, models, optimizers, preprocessing

# Directories and paths
home_dir = os.path.abspath(os.curdir)
log_path = os.path.join(home_dir, 'Codes', 'helper', 'base_log.log')
model_path = os.path.join(home_dir, 'Codes', 'A', 'A_lstm_model.h5')

# The configuration of logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w')


class A:
    def __init__(self, path):
        self.df = pd.read_csv(path, sep='\t', header=None)
        logging.info(f'[Dataset Location]: {path}')

    def calculate_time(self, start_time, end_time):
        """Calculate the execution time and print into the logfile

        Args:
            start_time (Time): the time when the scripts start running
            end_time (Time): the time when the scripts end

        Returns:

        """
        elapsed_time = end_time - start_time
        logging.info(
            f'[Execution time]: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    # # ====================================================================================================================
    # # Functions for data preprocessing
    def preprocess_data(self):
        """Preprocess the raw dataset, tokenize the tweets and split the dataset for model training

        Args:

        Returns:
            X_train, X_test, y_train, y_test: training and testing set

        """
        logging.info('==========[Data preprocessing begins]==========')
        start_time = time.time()

        # Preprocess the raw dataframe
        df = self.df.drop(columns=[0, 3])  # drop 0:id and 3:NaN
        df = df.rename(columns={1: "label", 2: "tweet"})
        df['label'].replace({"negative": 0, "positive": 1,
                             "neutral": 2}, inplace=True)
        logging.debug(f'[Preprocessed dataframe]: {df.head(5)}')

        # Transform the tweets into tokenized contents
        words_set, df_sentenses = self.tweet2words(df)
        df[['content']] = df_sentenses[["content"]]
        tokenizer = preprocessing.text.Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(df['content'])
        logging.debug(f'[Tokenized contents]: {df.head(5)}')

        X = tokenizer.texts_to_sequences(df['content'])
        X = preprocessing.sequence.pad_sequences(
            X, maxlen=200, padding='post', truncating='post')
        Y = df['label']
        logging.debug(f'[Features]: {X}')

        # Split the dataset into 80% training set and 20% testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=True, random_state=2021)

        end_time = time.time()
        self.calculate_time(start_time, end_time)
        logging.info('==========[Data preprocessing ends]==========')

        return X_train, X_test, y_train, y_test

    def tweet2words(self, df):
        """Transform the tweets into tokenized contents

        Args:
            df (Dataframe): the raw dataset

        Returns:
            words_set (Set): the set of words for replacing tokens extracted from the tweets
            df_sentenses (Dataframe): tokenized contents

        """
        words_set = set()  # use set to de-duplicate
        sentenses = []

        # Functions like TextPreProcessor() in the library ekphrasis was developed as part of the text processing pipeline for DataStories team's submission for SemEval-2017 Task 4 (English), Sentiment Analysis in Twitter
        text_processor = TextPreProcessor(normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
                                          annotate={
                                              'hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'},
                                          fix_html=True,
                                          segmenter='twitter',
                                          corrector='twitter',

                                          unpack_hashtags=True,
                                          unpack_contractions=True,
                                          spell_correct_elong=False,

                                          tokenizer=SocialTokenizer(
                                              lowercase=True).tokenize,
                                          dicts=[emoticons])

        for index in range(df.shape[0]):
            tweet = df['tweet'][index]
            token = text_processor.pre_process_doc(tweet)
            sentenses.append(' '.join(token))
            words_set.update(token)

        df_sentenses = pd.DataFrame(sentenses, columns=['content'])
        return words_set, df_sentenses

    # # ====================================================================================================================
    # # Functions for model training
    def build_model_lstm(self):
        """Build an LSTM (Long Short Term Memory) model

        Args:

        Returns:
            model: the model itself

        """
        words_size = 20000

        # Create a sequential model
        model = models.Sequential()

        model.add(layers.Embedding(words_size, 200))
        model.add(layers.GaussianNoise(0.1))
        model.add(layers.Dropout(0.1))

        # model.add(layers.LSTM(32)) # tested
        model.add(layers.Bidirectional(layers.LSTM(32)))
        model.add(layers.Dropout(0.1))

        # Output Layer (3 classes for sentiments)
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        # Display summary of the model
        model.summary()

        # Compile the model using ADAM (Adaptive learning rate optimization)
        model.compile(optimizer=optimizers.Adam(1e-4),
                      loss=losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        return model

    def callback(self):
        """Save the weights of the model into a local file when the best performance is achieved

        Args:

        Returns:
            callback_list (List): a list of EarlyStopping and ModelCheckpoint objects
        """
        # Seek a minimum for validation loss and display the stopped epochs using verbose and adding delays
        es = EarlyStopping(monitor='val_loss', mode='auto',
                           verbose=1, patience=3)

        # Save the best model using checkpoint
        mcp = ModelCheckpoint(os.path.normcase(
            model_path), monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

        # Define callback function in a list
        callback_list = [es, mcp]

        return callback_list

    def train_model_lstm(self, X_train, X_test, y_train, y_test, loaded_model=False):
        """Train an LSTM (Long Short Term Memory) model

        Args:
            X_train, X_test, y_train, y_test: training and testing set
            loaded_model (boolean): train a new model from scratch by default, load the existing trained model if marked as True

        Returns:
            train_acc: the prediction accuracy on the training set
            test_acc: the prediction accuracy on the testing set
        """

        logging.info('==========[Model training begins]==========')
        start_time = time.time()

        if loaded_model:
            model = models.load_model(model_path)
        else:
            model = self.build_model_lstm()
            cb_list = self.callback()

            num_epochs = 30  # 30
            val_split = 0.2
            batch_size = 128  # 32, 64 tested

            hist = model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size,
                             epochs=num_epochs, validation_split=val_split, shuffle=True, callbacks=cb_list)

            ##### for ploting only #####
            fig, axes = plt.subplots(1, 2, figsize=(20, 5))
            self.plot_learning_curve("Learning Curve(Bi-LSTM)",
                                     hist.history['accuracy'],
                                     hist.history['loss'],
                                     hist.history['val_accuracy'],
                                     hist.history['val_loss'],
                                     hist.epoch[-1] + 1,
                                     axes, ylim=(0.0, 1.01))

            # y_pred = model.predict(X_test)
            # logging.debug(f'[y_test]: {y_test}')
            # logging.debug(f'[y_pred]: {y_pred}')
            # labels = ['Neural', 'Positive', 'Negative']
            # self.plot_confusion_matrix(
            #     np.argmax(y_test), np.argmax(y_pred), labels, normalize='true')

            # logging.debug(
            #     f'[Classification report]: {classification_report(np.argmax(y_test),np.argmax(y_pred))}')

        train_loss, train_acc = model.evaluate(
            np.array(X_train), np.array(y_train), verbose=0)
        test_loss, test_acc = model.evaluate(
            np.array(X_test), np.array(y_test), verbose=0)

        end_time = time.time()
        self.calculate_time(start_time, end_time)
        logging.info('==========[Model training ends]==========')

        return train_acc, test_acc

    # # ====================================================================================================================
    # # Functions for visualization
    def plot_learning_curve(self, title, train_val, train_loss, val_val, val_loss, num_epoch, axes=None, ylim=None):
        """Plot learning curve
        """
        logging.debug(f'[num_epoch]: {num_epoch}')
        num = np.linspace(1, num_epoch, num_epoch)
        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xticks(num)
        axes[0].set_xlabel("Number of epochs")
        axes[0].set_ylabel("Score")

        axes[0].grid()
        axes[0].plot(num, train_val, 'o-', color="r",
                     label="Training score")
        axes[0].plot(num, val_val, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        axes[1].grid()
        axes[1].plot(num, train_loss, 'o-', color="r",
                     label="Training loss")
        axes[1].plot(num, val_loss, 'o-', color="g",
                     label="Cross-validation loss")
        axes[1].legend(loc="best")
        axes[1].set_title("Training Loss v.s. Validation Loss")
        axes[1].set_xticks(num)
        axes[1].set_xlabel("Number of epochs")
        axes[1].set_ylabel("Loss")

        plt.savefig('a_learning curve')
        # plt.show()
        return plt

    def plot_confusion_matrix(self, y_test, y_pred, labels, normalize=None):
        """Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred, normalize=normalize)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, cmap=plt.cm.get_cmap('Blues', 6))
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels, rotation=45)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.savefig('a_confusion matrix')
        # plt.show()
        return plt
