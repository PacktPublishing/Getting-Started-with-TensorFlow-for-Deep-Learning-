# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def get_data():
    iris   = datasets.load_iris()
    features = iris["data"]
    labels = iris["target"]

    one_hot = np.zeros(shape=(len(labels), 3))
    # for classification we need the labels to be in one hot vectors
    for i in range(0, len(labels)):
        one_hot[i, labels[i]] = 1
    return train_test_split(features, one_hot, test_size=0.33, random_state=RANDOM_SEED)

def main():
    train_feats, test_feats, train_lab, test_lab = get_data()

    # Layer's sizes
    feat_shape = train_feats.shape[1]  
    hidden_nodes = 10                   
    out_shape = train_lab.shape[1]   

    #define the graph
    graph = tf.Graph()
    with graph.as_default():
        # placeholders
        inputs = tf.placeholder("float", shape=[None, feat_shape])
        outputs = tf.placeholder("float", shape=[None, out_shape])

        #hidden layer
        W1 = tf.get_variable(name="W1", 
                                shape=[feat_shape, hidden_nodes], 
                                initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1", 
                                shape=[hidden_nodes], 
                                initializer=tf.constant_initializer(0.0))
        H1 = tf.matmul(inputs, W1) + b1
        H1 = tf.nn.relu(H1)

        # output
        W2 = tf.get_variable(name="W2", 
                                shape=[hidden_nodes, out_shape], 
                                initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2", 
                                shape=[out_shape], 
                                initializer=tf.constant_initializer(0.0))

        pred_tensor = tf.matmul(H1, W2) + b2
        predict = tf.argmax(pred_tensor, axis=1)

        # for training
        cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=pred_tensor))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # train
    with tf.Session(graph = graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(100):
            # Train with each example
            for i in range(len(train_feats)):
                op, cst = sess.run([updates, cost], feed_dict={inputs: train_feats[i: i + 1], outputs: train_lab[i: i + 1]})

            test_accuracy  = np.mean(np.argmax(test_lab, axis=1) ==
                                    sess.run(predict, feed_dict={inputs: test_feats, outputs: test_lab}))

            print("Epoch: %d, acc: %.2f, cost: %.5f"
                % (epoch, test_accuracy, cst))


if __name__ == '__main__':
    main()