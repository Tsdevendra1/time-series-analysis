import numpy as np
import tensorflow as tf
from generate_data import generate_data
import matplotlib.pyplot as plt

for learning_rate in [1, 1E-1,1E-2, 1E-3, 1E-4, 1E-5]:

    hparam_str = "{}".format(learning_rate)

    tf.reset_default_graph()
    writer = tf.summary.FileWriter('scripts/' + hparam_str)

    n_hidden = 150
    n_outputs = 50  # How much of a window we want to be predicted
    n_inputs = 1  # Number of inputs for each time step
    n_timesteps = 100
    EPOCHS = 150
    n_examples_training = 256
    n_examples_test = 3
    BATCH_SIZE = 64

    # Generate training and test data
    training_time_x, training_x, training_time_y, training_y = generate_data(n_examples_training, n_timesteps, n_outputs)
    training_x = training_x.reshape(n_examples_training, n_timesteps, n_inputs)
    test_time_x, test_x, test_time_y, test_y = generate_data(n_examples_test, n_timesteps, n_outputs, freq=1)
    variable = test_x
    test_x = test_x.reshape(n_examples_test, n_timesteps, n_inputs)

    # Create variables
    weights = tf.get_variable("weights", initializer=tf.random_normal([n_hidden, n_outputs]))
    bias = tf.get_variable("bias", initializer=tf.random_normal([n_outputs]))

    features = tf.placeholder(tf.float32, [None, n_timesteps, n_inputs])
    labels = tf.placeholder(tf.float32, [None, n_outputs])
    batch_size = tf.placeholder(tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    x, y = iterator.get_next()

    with tf.name_scope("LSTM") as scope:
        cell = tf.contrib.rnn.LSTMCell(n_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0,
                                         2])  # Change dimensions of outputs from [batch_size, timesteps, n_inputs] -> [timesteps, batch_size, n_inputs]
        last = tf.gather(outputs, int(outputs.get_shape()[
                                          0]) - 1)  # Take slice along the first axis. In this case we want output of last timestep = no. timesteps - 1 due to how indexing works

        # Note that the following gives same output as 'last' without all the hardwork of
        # transpose and gather (shown for information). (Can be checked by last-last_ = 0)
        last_ = states.h

    with tf.name_scope("prediction") as scope:
        # Dimentions of last = [no.examples, num_hidden], weight = [num_hidden, n_outputs] -> [no.examples, n_outputs]
        # Get prediction
        prediction = tf.nn.bias_add(tf.matmul(last_, weights), bias)  # bias_add broadcasts biases

    with tf.name_scope("Optimise") as scope:
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(prediction, y)))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(cost)
        tf.summary.scalar('cross_entropy', cost)

    # Initialise all variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        merged_summary = tf.summary.merge_all()
        writer.add_graph(sess.graph)
        sess.run(init)
        cost_epoch_counter = []  # To keep track of cost every epoch
        epoch = []  # Keep track of corresponding epoch
        for i in range(EPOCHS):
            sess.run(iterator.initializer, feed_dict={features: training_x, labels: training_y, batch_size: BATCH_SIZE})
            while True:
                try:
                    _, cost_epoch, random = sess.run([train, cost, merged_summary])
                except tf.errors.OutOfRangeError:
                    break
            if i % 1 == 0:
                print("EPOCH {}".format(i), cost_epoch)
                cost_epoch_counter.append(cost_epoch)
                epoch.append(i)
            writer.add_summary(random, i)


        plt.plot(epoch, cost_epoch_counter)
        plt.show(block=False)

        sess.run(iterator.initializer, feed_dict={features: test_x, labels: test_y, batch_size: n_examples_test})
        
        # Get prediction for test set
        prediction_test = sess.run(prediction)
     
        subplot_figure = plt.figure(1)
        plt.subplot(311)
        plt.plot(test_time_x[0,:].T, variable[0,:].T,test_time_y[0,:].T, prediction_test[0,:].T)
        plt.subplot(312)
        plt.plot(test_time_x[1,:].T, variable[1,:].T,test_time_y[1,:].T, prediction_test[1,:].T)
        plt.subplot(313)
        plt.plot(test_time_x[2,:].T, variable[2,:].T,test_time_y[2,:].T, prediction_test[2,:].T)
        plt.show(block=False)
        graph_name = 'graphs/' + hparam_str + '.png'
        subplot_figure.savefig(graph_name)  # save the figure to file
        plt.close(subplot_figure)  # close the figure

        writer.close()
