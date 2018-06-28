from __future__ import absolute_import, division, print_function

import base64
import os
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from six.moves import urllib
from six.moves.urllib.request import urlretrieve


def check_value(var, var_b64, var_name='var'):
    with open('{}.npy'.format(var_name), 'wb') as of:
        of.write(base64.b64decode(data))
    var_orig = np.load('{}.npy'.format(var_name), allow_pickle=False)
    diff = np.mean(np.abs(var_orig - var))
    assert diff < 1e-4, 'value[{}] mismatch [{}]'.format(var_name, diff)


# %matplotlib inline
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

image = tf.Variable(tf.zeros((299, 299, 3)))


def network(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_points = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_points


logits, probs, end_points = network(image, reuse=False)

#data_dir = tempfile.mkdtemp()
data_dir = '.'
checkpoint_filename = os.path.join(data_dir, 'inception_v3.ckpt')
if not os.path.exists(checkpoint_filename):
    #inception_tarball, _ = urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    inception_tarball = 'C:\\Users\\LuBu.LUBU-001\\Desktop\\inception_v3_2016_08_28.tar.gz'
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)


restore_vars = [
    var for var in tf.global_variables() if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.chpt'))


def get_feature(img, feature_layer_name):
    p, feature_values = sess.run([probs, end_points], feed_dict={image: img})
    return feature_values[feature_layer_name].squeeze()

#img_path  = os.path.join(data_dir, '1.jpg')
image_urls = [os.path.join(data_dir, '1.jpg')]

plt.figure(flgsize=(12, 12))
images = []
for idx, img_url in enumerate(image_urls):
    #img_path, _ = rulretrieve(image_url)
    img_path = img_url
    img = PIL.Image.open(img_path)
    img = img.resize((299, 299))

    plt.subplot(1, 8, idx+1)
    plt.axis('off')
    plt.imshow(img)
    plt.title('images[{}]'.format(idx))
    images.append(img)

layer = 'PreLogits'
features = []
for img in images:
    img = (np.asarray(img)/255.0).astype(np.float32)
    feature = get_feature(img, layer)
    features.append(feature)
feature_vectors = np.stack(features)

score_feature_shape = 0
try:
    print(feature_vectors.shape)
    assert feature_vectors.shape == (7, 2048), 'shape mismatch!'
    score_feature_shape = 10
except Exception as ex:
    print(ex)


np.save('feature.npy', feature_vectors, allow_pickle=False)
import base64

with open('feature.npy','rb') as inf:
    data= inf.read()
    print(base64.b64encode(data))

# score_feature_value = 0
# try:
#     data = b'sdjfsadfjl' # the data value from the inf.read() base64 data for picture 
#     check_value(feature_vectors, data,'feature_vectors')
#     score_feature_value =30
# except Exception as ex:
#     print(ex)
