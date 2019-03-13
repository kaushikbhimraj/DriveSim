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
Beta   = np.array([line.split(',')[0] for line in open(path) if line[0:1] != '\n'])
Delta  = np.array([line.split(',')[1] for line in open(path) if line[0:1] != '\n'])
Target = np.array([line.split(',')[2] for line in open(path) if line[0:1] != '\n'])

# Converting Target to 0 or 1 format.
Target = np.array([1 if y == "Distracted\n" else 0 for y in Target])

# Spliting the datasets into training and testing portions
sess = tf.Session()
train_indices = np.random.choice(len(Beta), round(len(Beta) * 0.8), replace=False)
test_indices  = np.array(list(set(range(len(Beta))) - set(train_indices)))

Beta_train    = Beta[train_indices]
Beta_test     = Beta[test_indices]
Target_train  = Target[train_indices]
Target_text   = Target[test_indices]

batch_size    = 100
x_data        = tf.placeholder(shape = [None, 2], dtype=tf.float32)
y_target      = tf.placeholder(shape = [None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

model_output = tf.subtract(tf.matmul(x_data, A), b)

l2_norm  = tf.reduce_sum(tf.square(A))
alpha    = tf.constant([0.1])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))

loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
prediction = tf.sign(model_output)
accuracy   = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

my_opt     = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.