# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import tensorflow as tf


FLAGS = None


def main(_):
  # Import data
    train = pd.read_csv('D:/./iris_training.csv')
    test  = pd.read_csv('D:/./iris_test.csv')
      # Create the model
    train_ohe = pd.get_dummies(train['Species'])
    f= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    train_ohe = pd.concat([train[f],train_ohe],axis=1)


    test_ohe = pd.get_dummies(test['Species'])
    test_ohe = pd.concat([test[f],test_ohe],axis=1)

    #print(train_ohe.head())

    train_id = train_ohe.filter(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    train_ld = train_ohe.filter([0,1,2])
    test_id = test_ohe.filter(['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
    test_ld = test_ohe.filter([0,1,2])
    x = tf.placeholder(tf.float32, [None,4 ])
    W = tf.Variable(tf.zeros([4,3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, W) + b

      # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 3])

      # The raw formulation of cross-entropy,
      #
      #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
      #                                 reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
      # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
     # Test trained model
    
      # Train
    for _ in range(1000):
        #batch_xs, batch_ys = next_batch(20,mnist,)
       
       sess.run(train_step, feed_dict={x: train_id, y_: train_ld})

      # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy for Iris: ",sess.run(accuracy, feed_dict={x: test_id, y_: test_ld}))
    



if __name__ == '__main__':
    """parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/iris/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()"""
    tf.app.run(main=main)
