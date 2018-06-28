import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# weight
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

#logits and output
logits = tf.matmul(x, w) + b
output = tf.nn.sigmoid(logits)
# cross_entropy
cross_entropy = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=y, logits=logits)
# train_step
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

x_value = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_value = np.array([[1], [1], [1], [0]])

init_op = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_op)

# cross_entropy_value, logits_value, output_value = sess.run(
#     [cross_entropy, logits, output], feed_dict={x: x_value, y: y_value})


# for current_step in range(100):
#     cross_entropy_value, logits_value, output_value,_ = sess.run(
#         [cross_entropy, logits, output,train_step], feed_dict={x: x_value, y: y_value})


for current_step in range(100):
    cross_entropy_value, logits_value, output_value, _, w_value, b_value = sess.run(
        [cross_entropy, logits, output, train_step, w, b], feed_dict={x: x_value, y: y_value})


print(cross_entropy_value)
print(logits_value)
print(output_value)
print(w_value)
print(b_value)
