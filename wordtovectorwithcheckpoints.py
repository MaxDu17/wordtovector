from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    with tf.name_scope('data'):
        center_index = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name = "center_index")
        target_index = tf.placeholder(tf.int32, shape = [BATCH_SIZE,1],name="target_index")

    with tf.name_scope('embed'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE],-1.0,1.0),name = "embedded_matrix")

        embed = tf.nn.embedding_lookup(embed_matrix, center_index, name='embed')#embed is the word vector

    with tf.name_scope('loss'):
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],stddev=1.0/EMBED_SIZE**0.5),name="nce_weight")
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]),name="nce_bias")
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,biases=nce_bias,labels=target_index,inputs=embed,num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE),name="nce_loss_function")
        #it feeds in the word vector, and compares it to the target index, which is what the word should be, which is an index.
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step = global_step)#this is separate
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("histogram loss", loss)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('my_checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/withcheckpoints/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            batch = next(batch_gen)
            loss_batch, _ = sess.run([loss,optimizer], feed_dict = {center_index: batch[0], target_index: batch[1]})# the "_" just acts like a dump ground for "optimizer"
            summary = sess.run(summary_op, feed_dict = None)
            writer.add_summary(summary,global_step = global_step)
            total_loss = total_loss + loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                saver.save(sess, "my_checkpoints/wordtovector", global_step = global_step)
                total_loss = 0.0
        writer.close()


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)


if __name__ == '__main__':
    main()