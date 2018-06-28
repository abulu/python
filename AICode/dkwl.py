from __future__ import absolute_import, division, print_function

import json
import os
import tarfile

import numpy as np
import PIL
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from matplotlib import pyplot as plt
from six.moves import urllib
from six.moves.urllib.request import urlretrieve
from tensorflow.examples.tutorials.mnist import input_data

# %matplotlib inline

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

# 声明一个可训练的变量（varialbe），作为网络的输入数据。
image = tf.Variable(tf.zeros((299, 299, 3)))
# 从slim中调用inception的网络定义。这里我们设置is_training=False，避免dropout。


def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs


logits, probs = inception(image, reuse=False)

# 下载预训练的inception_v3的checkpoint。
data_dir = '.'
checkpoint_filename = os.path.join(data_dir, 'inception_v3.ckpt')
if not os.path.exists(checkpoint_filename):
    inception_tarball, _ = urlretrieve(
        'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

# 将预训练权重恢复到模型中。
restore_vars = [
    var for var in tf.global_variables() if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))
# 下载一个imagenet的类别标签文本，用于显示。

imagenet_filename = os.path.join(data_dir, 'imagenet.json')
if not os.path.exists(imagenet_filename):
    imagenet_json, _ = urlretrieve(
        'http://www.anishathalye.com/media/2017/07/25/imagenet.json')
    with open(imagenet_json) as f:
        imagenet_labels = json.load(f)
else:
    with open(imagenet_filename) as f:
        imagenet_labels = json.load(f)

# 对图片进行分类，并将结果可视化。
# norm the image for imshow to avoid a bug in matplotlib


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def classify(img, correct_class=None, target_class=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.sca(ax1)

    # run inference on image
    p = sess.run(probs, feed_dict={image: img})[0]

    img_norm = norm(img)
    ax1.imshow(img_norm)
    ax1.axis('off')
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(
        range(10), [imagenet_labels[i][:15] for i in topk],
        rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()

# 原始图片的下载和预处理。 基于inception_v3的预处理方法，图片缩放到299x299，并将数据转换为浮点数，映射到0-1区间。
# 由于这里的图片正好是方形的，所以不需要考虑缩放比例的问题。


img_path, _ = urlretrieve(
    'http://www.anishathalye.com/media/2017/07/25/cat.jpg')
#img_path  = os.path.join(data_dir, '1.jpg')
#img_path  = os.path.join(data_dir, '2.jpg')
img_class = 281
img = PIL.Image.open(img_path)
img = img.resize((299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)
# 对图片进行分类和可视化。
classify(img, correct_class=img_class)

# 声明一个x和 x̂ x^ ,其中x是原始图片，而 x̂ x^ 是可训练的对抗样本。当我们使用梯度下降对loss进行优化时，实际上优化的是 x̂ x^ 。初始化的时候，我们让 x̂ =xx^=x 。


x = tf.placeholder(tf.float32, (299, 299, 3))

x_hat = image  # our trainable adversarial input
assign_op = tf.assign(x_hat, x)

# ŷ y^ 是我们想要伪造成的目标分类。交叉熵按照ground truth= ŷ y^ 来计算。
# 使用sparse_softmax_cross_entropy_with_logits能省略one_hot的操作，减少一些麻烦。

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=[y_hat])

# 注意这里的var_list，限定仅优化x_hat
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])

# 为了让对抗样本和原始数据看上去差不多，我们约束 x̂ x^ 的每个像素和原始数据对应像素的值尽可能一致，差距小于epsilon（自己指定的超参数）

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

# 通过梯度下降方式对 x̂ x^ 进行优化修正，每次优化确保和原始图像偏差幅度不超过epsilon。
demo_epsilon = 2.0/255.0  # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 947  # "mushroom"

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    #sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i+1) % 10 == 0:
        print('step %d, loss=%g' % (i+1, loss_value))

# output the training step

adv = x_hat.eval()  # retrieve the adversarial example
classify(adv, correct_class=img_class, target_class=demo_target)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.imshow(img)
ax1.axis('off')
ax1.set_title('img')
ax2.imshow(norm(adv))
ax2.axis('off')
ax2.set_title('adv')

diff = img - adv

ax3.imshow(norm(diff))
ax3.axis('off')
ax3.set_title('diff')

plt.show()
