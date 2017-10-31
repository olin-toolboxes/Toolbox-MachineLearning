"""Explore learning curves for classification of handwritten digits"""

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    """Read in the 8x8 pictures of numbers and display 10 of them"""
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    """Train a model on pictures of digits.

    Read in 8x8 pictures of numbers and evaluate the accuracy of the model
    when different percentages of the data are used as training data. This function
    plots the average accuracy of the model as a function of the percent of data
    used to train it.
    """
    data = load_digits()
    num_trials = 30
    train_percentages = list(range(5, 95, 5))
    test_accuracies = numpy.zeros(len(train_percentages))

    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner

    for i in range(0, len(train_percentages)):
        score_sum = 0
        for n in range(0, num_trials):
            x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,
            train_size = train_percentages[i]/100)
            model = LogisticRegression(C=10**-1)
            model.fit(x_train, y_train)
            score_sum += model.score(x_test, y_test)
        score = score_sum / n
        test_accuracies[i] = score

    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()

if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    #display_digits()
    train_model()
    #be_model()
