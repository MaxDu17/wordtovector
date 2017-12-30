from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000  # how many steps to skip before reporting the loss


def word2vec(batch_gen):

    center_index = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    target_index = tf.laceholder(tf.int32, shape = [BATCH_SIZE])

    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE],-1.0,1.0))


    embed = tf.nn.embedding_lookup(embed_matrix, center_index, name='embed')#embed is the word vector

    # TO DO

    # Step 4: construct variables for NCE loss
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
    # bias: vocab size, initialized to 0

    # TO DO

    # define loss function to be NCE loss function
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # need to get the mean accross the batch
    # note: you should use embedding of center words for inputs, not center words themselves

    # TO DO

    # Step 5: define optimizer

    # TO DO

    with tf.Session() as sess:
        # TO DO: initialize variables

        total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch

            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)


if __name__ == '__main__':
    main()