import tensorflow as tf
from process_data import process_data
from wordtovectorlib import *

class Hyperparameters:
    VOCAB_SIZE = 50000
    BATCH_SIZE = 128
    EMBED_SIZE = 128  # dimension of the word embedding vectors
    SKIP_WINDOW = 1  # the context window
    NUM_SAMPLED = 64  # Number of negative examples to sample.
    LEARNING_RATE = 1.0
    NUM_TRAIN_STEPS = 20000
    SKIP_STEP = 2000  # how many steps to skip before reporting the loss

hyp = Hyperparameters()

batch_gen = process_data(hyp.VOCAB_SIZE, hyp.BATCH_SIZE, hyp.SKIP_WINDOW)
center_index, target_index = initialize_vars(hyp)
embed = initialize_embed(hyp, center_index)
loss, optimizer = initialize_loss_and_optimizer(hyp,target_index,embed)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_loss = 0.0  # we use this to calculate the average loss in the last SKIP_STEP steps
    writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
    for index in range(hyp.NUM_TRAIN_STEPS):
        batch = next(batch_gen)
        loss_batch, _ = sess.run([loss,optimizer], feed_dict = {center_index: batch[0], target_index: batch[1]})# the "_" just acts like a dump ground for "optimizer"
        total_loss = total_loss + loss_batch
        if (index + 1) % hyp.SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index, total_loss / hyp.SKIP_STEP))
            total_loss = 0.0
    writer.close()


