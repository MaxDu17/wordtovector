import tensorflow as tf
import numpy as np
from process_data import process_data
def initialize_vars(hyp):
    with tf.name_scope('data'):
        center_index = tf.placeholder(tf.int32, shape=[hyp.BATCH_SIZE], name = "center_index")
        target_index = tf.placeholder(tf.int32, shape = [hyp.BATCH_SIZE,1],name="target_index")
        return center_index, target_index

def initialize_embed(hyp, center_index):
    with tf.name_scope('embed'):
        embed_matrix = tf.Variable(tf.random_uniform([hyp.VOCAB_SIZE, hyp.EMBED_SIZE], -1.0, 1.0), name="embedded_matrix")

        embed = tf.nn.embedding_lookup(embed_matrix, center_index, name='embed')  # embed is the word vector
        return embed

def initialize_loss_and_optimizer(hyp,target_index,embed):
    with tf.name_scope('loss'):
        nce_weight = tf.Variable(tf.truncated_normal([hyp.VOCAB_SIZE, hyp.EMBED_SIZE],stddev=1.0/hyp.EMBED_SIZE**0.5),name="nce_weight")
        nce_bias = tf.Variable(tf.zeros([hyp.VOCAB_SIZE]),name="nce_bias")
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,biases=nce_bias,labels=target_index,inputs=embed,num_sampled=hyp.NUM_SAMPLED, num_classes=hyp.VOCAB_SIZE),name="nce_loss_function")
        #it feeds in the word vector, and compares it to the target index, which is what the word should be, which is an index.
    optimizer = tf.train.GradientDescentOptimizer(hyp.LEARNING_RATE).minimize(loss)#this is separate
    return loss, optimizer

