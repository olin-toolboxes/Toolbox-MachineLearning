"""Explore learning curves for classification of handwritten digits"""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    """Read in the 8x8 pictures of numbers and display 10 of them."""
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i + 1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()

def train_model():
    """Train a model on pictures of digits.

    Read in 8x8 pictures of numbers and evaluate the accuracy of the model
    when different percentages of the data are used as training data. This
    y_size plots the average accuracy of the model as a function of the percent
    of data used to train it.
    """
    data = load_digits()
    num_trials = 70
    train_percentages = range(5, 95, 5)
    test_accuracies = numpy.zeros(len(train_percentages))

    for i in range(0, len(train_percentages)):
        total_model_score = 0
        for n in range(0, num_trials):
            x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
            train_size = train_percentages[i]/100)
            model = LogisticRegression(C=10**-1) #change to C=10**-10 for question 4
            model.fit(x_train, y_train)
            total_model_score += model.score(x_test, y_test)
        test_accuracies[i] = total_model_score/n

    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel("Percentage of Data Used for Training")
    plt.ylabel("Accuracy on Test Set")
    plt.show()


if __name__ == "__main__":
    # display_digits()
    train_model()
