#!/usr/bin/env python
# -*- coding: utf-8 -*-

u"""Chainer Tutorial — Chainer 1.0.0 documentation
http://docs.chainer.org/en/latest/tutorial/index.html

Example: Multi-layer Perceptron on MNIST
"""

import argparse
import logging
import numpy as np
import scipy as sp
from sklearn.datasets import fetch_mldata
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

verbose = False
debug = False
logger = None

def init_logger():
    global logger
    logger = logging.getLogger('MNIST')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def main(args):
    def forward(x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        h1 = F.relu(model.l1(x))  # activation function
        h2 = F.relu(model.l2(h1)) # ReLU does not have parameters to optimize
        y = model.l3(h2)
        # the loss function of softmax regression
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)  # current accuracy

    def evaluate():
        sum_loss, sum_accuracy = 0, 0
        for i in xrange(0, 10000, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]
            loss, accuracy = forward(x_batch, y_batch)
            sum_loss += loss.data * batchsize
            sum_accuracy += accuracy.data * batchsize
        mean_loss = sum_loss / 10000
        mean_accuracy = sum_accuracy / 10000
        print(mean_loss[0], mean_accuracy)
        return

    global debug, verbose
    debug = args.debug
    if debug == True:
        verbose = True
    else:
        verbose = args.verbose

    mnist = fetch_mldata('MNIST original')
    x_all = mnist.data.astype(np.float32) / 255  # Scaling features to [0, 1]
    y_all = mnist.target.astype(np.int32)
    x_train, x_test = np.split(x_all, [60000])   # 60000 for training, 10000 for test
    y_train, y_test = np.split(y_all, [60000])

    # Simple three layer rectfier network
    model = FunctionSet(
        l1 = F.Linear(784, 100),  # 784 pixels -> 100 units
        l2 = F.Linear(100, 100),  # 100 units -> 100 units
        l3 = F.Linear(100, 10),   # 100 units -> 10 digits
    )
    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())

    batchsize = 100
    for epoch in xrange(20):
        if verbose: logger.info('epoch: {}'.format(epoch))
        indexes = np.random.permutation(60000)
        for i in xrange(0, 60000, batchsize):
            x_batch = x_train[indexes[i:i+batchsize]]
            y_batch = y_train[indexes[i:i+batchsize]]

            optimizer.zero_grads()  # Initialize gradient arrays
            loss, accuracy = forward(x_batch, y_batch)  # loss function
            loss.backward()  # Backpropagation
            optimizer.update()
        evaluate()

    return 0


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
