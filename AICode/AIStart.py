import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
#%matplotlib inline

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("./")

# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)

# print(mnist.validation.images.shape)
# print(mnist.validation.labels.shape)

# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)

plt.figure(figsize=(8,8))

for idx in range(16):
    plt.subplot(4,4, idx+1)
    plt.axis('off')
    plt.title('[{}]'.format(mnist.train.labels[idx]))
    plt.imshow(mnist.train.images[idx].reshape((28,28)))
    
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("int64", [None])
learning_rate = tf.placeholder("float")

def initialize(shape, stddev=0.1):
  return tf.truncated_normal(shape, stddev=0.1)

L1_units_count = 200

W_1 = tf.Variable(initialize([784, L1_units_count], stddev=0.05))
b_1 = tf.Variable(initialize([L1_units_count]))
logits_1 = tf.matmul(x, W_1) + b_1
output_1 = tf.nn.relu(logits_1)

L1_1_units_count = 100
W_1_1 = tf.Variable(initialize([L1_units_count,L1_1_units_count],stddev=0.06))
b_1_1 = tf.Variable(initialize([L1_1_units_count]))
logits_1_1 = tf.matmul(output_1,W_1_1)+b_1_1
output_1_1 = tf.nn.relu(logits_1_1)

L2_units_count = 10 
W_2 = tf.Variable(initialize([L1_1_units_count, L2_units_count], stddev=0.063))
b_2 = tf.Variable(initialize([L2_units_count]))
logits_2 = tf.matmul(output_1_1, W_2) + b_2  

logits = logits_2

cross_entropy_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy_loss)

pred = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(pred, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

batch_size = 32
trainig_step = 2000

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #定义验证集与测试集
    validate_data = {
        x: mnist.validation.images,
        y: mnist.validation.labels,
    }
    test_data = {x: mnist.test.images, y: mnist.test.labels}

    for i in range(trainig_step):
        xs, ys = mnist.train.next_batch(batch_size)
        _, loss = sess.run(
            [optimizer, cross_entropy_loss],
            feed_dict={
                x: xs,
                y: ys,
                learning_rate: 0.3
            })

        #每100次训练打印一次损失值与验证准确率
        if i > 0 and i % 100 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
            print(
                "after %d training steps, the loss is %g, the validation accuracy is %g"
                % (i, loss, validate_accuracy))
            saver.save(sess, './model.ckpt', global_step=i)

    print("the training is finish!")
    #最终的测试准确率
    acc = sess.run(accuracy, feed_dict=test_data)
    print("the test accuarcy is:", acc)

if validate_accuracy >=0.98:
    score = 100
elif validate_accuracy >=0.96 and validate_accuracy <0.98 :
    score = 60
else:
    score = 0

print('#'*10)
print('Your final score:[{}]'.format(score))
print('#'*10)