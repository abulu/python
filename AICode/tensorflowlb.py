import tensorflow as tf
import numpy as np
#print(tf.__version__)

a = tf.constant(0)
b = tf.constant(1)

# print(a)

c = tf.add(a, b)
d = a + b
e = a * b

# print(c)
# print(d)
# print(e)

mat_a = tf.constant([[1, 1, 1], [3, 3, 3]])
mat_b = tf.constant([[2, 2, 2], [5, 5, 5]], name='mat_b')

mul_a_b = mat_a * mat_b

tf_mul_a_b = tf.multiply(mat_a, mat_b)
tf_matmul_a_b = tf.matmul(mat_a, tf.transpose(mat_b), name='matmul_with_name')

this_graph = tf.get_default_graph()
this_graph_def = this_graph.as_graph_def()
print(this_graph_def)

sess = tf.Session()
mul_value, tf_mul_value, tf_matmul_value = sess.run(
    [mul_a_b, tf_mul_a_b, tf_matmul_a_b])
print(mul_value)
print(tf_mul_value)
print(tf_matmul_value)