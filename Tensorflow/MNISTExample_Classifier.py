
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import argparse
import sys

import itertools
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    IMAGE_TRAINING_DATA = mnist.train.images
    IMAGE_TRAINING_LABELS = mnist.train.labels

    IMAGE_TEST_DATA = mnist.test.images
    IMAGE_TEST_LABELS = mnist.test.labels

    IMAGE_VALIDATION_DATA = mnist.validation.images
    IMAGE_VALIDATION_LABELS = mnist.validation.labels

    def input_labels(data_set):
        labels = [];
        for set in data_set:
            i = 0
            for l in set:
                if l == 1:
                    labels.append(int(i));
                    break;
                i = i + 1;

        return labels

   
    optimizer = tf.train.ProximalGradientDescentOptimizer(0.5)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[300],
                                                n_classes=10,
                                                optimizer= optimizer,
                                                model_dir="/temp/mnist_model")

    def get_train_inputs():
        x = tf.constant(value=IMAGE_TRAINING_DATA,shape=[550000, 784], dtype=tf.float32)
        y = tf.constant(value=input_labels(IMAGE_TRAINING_LABELS), shape=[550000,1], dtype=tf.int32)

        return x,y

    classifier.fit(input_fn=get_train_inputs, steps=1);

    def get_test_inputs():
        x = tf.constant(value=IMAGE_TEST_DATA,shape=[100000, 784], dtype=tf.float32)
        y = tf.constant(value=input_labels(IMAGE_TEST_LABELS), shape=[100000,1], dtype=tf.int32)

        return x,y

    print('testing accuracy')
    accuracy = classifier.evaluate(input_fn=get_train_inputs, steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy))

    def get_prediction_inputs():
        x = tf.constant(value=IMAGE_VALIDATION_DATA, shape=[5000, 784], dtype=tf.float32)

        return x

    prediction = classifier.predict(input_fn=get_prediction_inputs);
    predictions = list(itertools.islice(prediction, 5000))
    
    for p in predictions:
        print(p);


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)