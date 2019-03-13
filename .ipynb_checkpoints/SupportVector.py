"""
Experimenting with support vector machines
"""
# Importing all the libraries 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Reading from the csv file.
path = r'/Users/KaushikBhimraj/Desktop/Merged_Data.csv'

# List comprehension used to create numpy arrays for inputs and targets.
BetaDelta   = np.array([[line.split(',')[0], line.split(',')[1]] for line in open(path) if line[0:1] != '\n'])
Target      = np.array([line.split(',')[2] for line in open(path) if line[0:1] != '\n'])

# Converting Target to 0 or 1 format.
Target      = np.array([1 if y == "Distracted\n" else 0 for y in Target])

# Spliting the datasets into training and testing portions
sess          = tf.Session()
train_indices = np.random.choice(len(BetaDelta), round(len(BetaDelta) * 0.8), replace=False)
test_indices  = np.array(list(set(range(len(BetaDelta))) - set(train_indices)))

Beta_train    = BetaDelta[train_indices]
Beta_test     = BetaDelta[test_indices]
Target_train  = Target[train_indices]
Target_text   = Target[test_indices]


# Here we want a very large batch size to achieve convergence. 
# The A variable will take on the 2 x 1 shape. (In the book, this is done because there are 2 predictor variables.)
batch_size    = 100
x_data        = tf.placeholder(shape = [None, 2], dtype=tf.float32)
y_target      = tf.placeholder(shape = [None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# For correctly classified points, values of >= 1 if target is 1.
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Calculate the L2 normm of a vector. 
# Margin parameter: a
l2_norm      = tf.reduce_sum(tf.square(A))
alpha        = tf.constant([0.1])

# Declaring classification loss and adding together the two terms.
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss                = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# Declaring the prediction and accuracy functions to evaluate bot the training and test sets.
prediction = tf.sign(model_output)
accuracy   = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Declaring optimization function and initializing model variables.
my_opt     = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init       = tf.global_variables_initializer()
sess.run(init)

# Starting the training loop. 
# Making sure to record loss and training accuracy for both training and test sets. 
loss_vec = []
train_accuracy = []
test_accuracy  = []

for i in range(500):
    rand_index = np.random.choice(len(Beta_train), size=batch_size)
    rand_x     = Beta_train[rand_index]
    rand_y     = np.transpose([Target_train[rand_index]])
    # x = sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    # y = sess.run(loss, feed_dict = {x_data:rand_x, y_target:rand_y})
