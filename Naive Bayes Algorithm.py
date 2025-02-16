"""Naive Bayes Classifiers."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import poisson
from collections import Counter


class NaiveBayesFilter(ClassifierMixin):
    """
    A Naive Bayes Classifier that sorts messages into spam or ham.
    """

    def __init__(self):
        """
        Initialize the NaiveBayesFilter.
        """
        self.P_spam = None
        self.P_ham = None
        self.spam_probs = None
        self.ham_probs = None

    # Problem 1
    def fit(self, X, y):
        """
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        """
        # Compute P(C=Ham) = Number of Ham messages / Total number of messages
        self.P_ham = np.sum(y == "ham") / len(y)
        # Compute P(C=Spam) = Number of Spam messages / Total number of messages
        self.P_spam = np.sum(y == "spam") / len(y)

        # Getting the list of all words in the training set
        words = " ".join(X).split()

        # Getting the vocabulary
        vocab = set(words)

        # Creating a dictionary to store the count of each word in spam messages
        spam_counts = {word: 0 for word in vocab}

        # Creating a dictionary to store the count of each word in ham messages
        ham_counts = {word: 0 for word in vocab}

        # Counting the number of times each word appears in spam and ham messages
        for message, label in zip(X, y):
            for word in message.split():
                if label == "spam":
                    spam_counts[word] += 1
                else:
                    ham_counts[word] += 1

        # Getting the total number of word occurrences in spam messages
        total_spam = sum(spam_counts.values())

        # Getting the total number of word occurrences in ham messages
        total_ham = sum(ham_counts.values())

        # Calculating P(xi | C = Spam) and P(xi | C = Ham) for each word
        self.spam_probs = {
            word: (spam_counts[word] + 1) / (total_spam + 2) for word in vocab
        }
        self.ham_probs = {
            word: (ham_counts[word] + 1) / (total_ham + 2) for word in vocab
        }

        return self

    # Problem 2
    def predict_proba(self, X):
        """
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        """
        # Initializing the log probabilities
        log_probs = []

        # Initializing the log probabilities for spam and ham
        log_prob_spam = np.log(self.P_spam)
        log_prob_ham = np.log(self.P_ham)

        # Iterating through each message
        for i, message in enumerate(X):
            # Initializing the spam and ham sums
            spam_sum = log_prob_spam
            ham_sum = log_prob_ham

            # Splitting the message into words
            words = message.split()

            # Iterating through each word in the message
            for word in words:
                # Checking if the word is in the vocabulary and if it isn't use 1/2 as the probability for spam and ham
                if word not in self.spam_probs:
                    self.spam_probs[word] = 1 / 2
                if word not in self.ham_probs:
                    self.ham_probs[word] = 1 / 2

                # Adding the log probability of the word given spam to the spam sum
                spam_sum += np.log(self.spam_probs[word])

                # Adding the log probability of the word given ham to the ham sum
                ham_sum += np.log(self.ham_probs[word])

            # Appending the log probabilities for the message
            log_probs.append([ham_sum, spam_sum])

        # Converting the log probabilities to a numpy array
        log_probs = np.array(log_probs)

        return log_probs

    # Problem 3
    def predict(self, X):
        """
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        """
        # Getting the log probabilities
        log_probs = self.predict_proba(X)

        # Initializing the predictions
        predictions = []

        # Iterating through each log probability
        for log_prob in log_probs:
            # Checking for a tie and appending ham if there is a tie
            if log_prob[0] == log_prob[1]:
                predictions.append("ham")
                continue
            # Appending the prediction based on the log probability
            predictions.append("spam" if log_prob[1] > log_prob[0] else "ham")

        return np.array(predictions)


def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.

    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # Splitting the data
    df = pd.read_csv("./Data/sms_spam_collection.csv")
    X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"])

    # Creating the NaiveBayesFilter
    nb = NaiveBayesFilter()

    # Fitting the NaiveBayesFilter
    nb.fit(X_train, y_train)

    # Getting the predictions
    predictions = nb.predict(X_test)

    # Getting the proportion of spam messages correctly identified
    spam_correct = np.sum((predictions == "spam") & (y_test == "spam")) / np.sum(
        y_test == "spam"
    )

    # Getting the proportion of ham messages incorrectly identified
    ham_incorrect = np.sum((predictions == "spam") & (y_test == "ham")) / np.sum(
        y_test == "ham"
    )

    return (spam_correct, ham_incorrect)


# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    """
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    """

    def __init__(self):
        """
        Initialize the PoissonBayesFilter.
        """
        self.P_spam = None
        self.P_ham = None
        self.spam_rates = {}
        self.ham_rates = {}
        self.spam_count = 0
        self.ham_count = 0

    def fit(self, X, y):
        """
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        """
        # Calculating P(C=Ham) and P(C=Spam)
        self.P_ham = np.sum(y == "ham") / len(y)
        self.P_spam = np.sum(y == "spam") / len(y)

        # Calculating the vocabulary
        words = " ".join(X).split()

        # Getting the vocabulary
        vocab = set(words)

        # Calculating the number of words in the spam class and the ham class
        for message, label in zip(X, y):
            if label == "spam":
                self.spam_count += len(message.split())
            else:
                self.ham_count += len(message.split())

        # Calculating r_{i,k} for each word
        for word in vocab:
            # Calculating the number of occurences of word in spam messages
            spam_word_count = 0
            for message, label in zip(X, y):
                if label == "spam":
                    spam_word_count += message.split().count(word)

            # Calculating the number of occurences of word in ham messages
            ham_word_count = 0
            for message, label in zip(X, y):
                if label == "ham":
                    ham_word_count += message.split().count(word)

            # Calculating spam_rates and ham_rates, each of which is a dictionary
            # with keys equal to the words in the vocabulary and values equal to the r values
            self.spam_rates[word] = (spam_word_count + 1) / (self.spam_count + 2)
            self.ham_rates[word] = (ham_word_count + 1) / (self.ham_count + 2)

        return self

    def predict_proba(self, X):
        """
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        """
        # Initializing the log probabilities
        log_P_ham = np.log(self.P_ham)
        log_P_spam = np.log(self.P_spam)

        # Initializing the log probabilities
        log_probs = []

        # Iterating through each message
        for message in X:
            # Splitting the message into words
            words = message.split()

            # Initializing the spam and ham sums
            spam_sum = log_P_spam
            ham_sum = log_P_ham

            # Getting the total number of words in the message
            total_words = np.unique(words, return_counts=True)[1].sum()

            # Iterating through each word in the message
            for word in words:
                # Checking if the word is in the vocabulary and if it isn't use 1 / (self.class_count + 2) as the probability for its respective class
                if word not in self.spam_rates:
                    self.spam_rates[word] = 1 / (self.spam_count + 2)
                if word not in self.ham_rates:
                    self.ham_rates[word] = 1 / (self.ham_count + 2)

                # Getting the number of times the word appears in the message
                word_counts = Counter(words)
                word_count = word_counts[word]

                # Adding the log probability of the word given spam to the spam sum using scipy.stats.poisson.logpmf
                spam_sum += poisson.logpmf(
                    word_count, mu=self.spam_rates[word] * total_words
                )

                # Adding the log probability of the word given ham to the ham sum using scipy.stats.poisson.logpmf
                ham_sum += poisson.logpmf(
                    word_count, mu=self.ham_rates[word] * total_words
                )

            # Appending the log probabilities for the message
            log_probs.append([ham_sum, spam_sum])

        # Converting the log probabilities to a numpy array
        log_probs = np.array(log_probs)

        return log_probs

    def predict(self, X):
        """
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        """
        # Getting the log probabilities
        log_probs = self.predict_proba(X)

        # Initializing the predictions
        predictions = []

        # Iterating through each log probability
        for log_prob in log_probs:
            # Checking for a tie and appending ham if there is a tie
            if log_prob[0] == log_prob[1]:
                predictions.append("ham")
                continue
            # Appending the prediction based on the log probability
            predictions.append("spam" if log_prob[1] > log_prob[0] else "ham")

        return np.array(predictions)


def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.

    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # Splitting the data
    df = pd.read_csv("./Data/sms_spam_collection.csv")
    X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"])

    # Creating the PoissonBayesFilter
    pb = PoissonBayesFilter()

    # Fitting the PoissonBayesFilter
    pb.fit(X_train, y_train)

    # Getting the predictions
    predictions = pb.predict(X_test)

    # Getting the proportion of spam messages correctly identified
    spam_correct = np.sum((predictions == "spam") & (y_test == "spam")) / np.sum(
        y_test == "spam"
    )

    # Getting the proportion of ham messages incorrectly identified
    ham_incorrect = np.sum((predictions == "spam") & (y_test == "ham")) / np.sum(
        y_test == "ham"
    )

    return (spam_correct, ham_incorrect)

def sklearn_naive_bayes(X_train, y_train, X_test):
    """
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    """
    # Creating a count vectorizer
    cv = CountVectorizer()

    # Transforming the training data
    X_train = cv.fit_transform(X_train)

    # Transforming the test data
    X_test = cv.transform(X_test)

    # Creating a multinomial naive bayes classifier
    nb = MultinomialNB()

    # Fitting the classifier
    nb.fit(X_train, y_train)

    # Getting the predictions
    predictions = nb.predict(X_test)

    return predictions


# Testing
# Importing the data
df = pd.read_csv("./Data/sms_spam_collection.csv")

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"])
