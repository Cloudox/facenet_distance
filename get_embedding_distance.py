# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet

image_size = 200 #don't need equal to real image size, but this value should not small than this
modeldir = './model_check_point/20170512-110547.pb' #change to your model dir
image_name1 = 'x.jpg' #change to your image name
image_name2 = 'y.jpg' #change to your image name

print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print('facenet embedding模型建立完毕')

scaled_reshape = []

image1 = scipy.misc.imread(image_name1, mode='RGB')
image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image1 = facenet.prewhiten(image1)
scaled_reshape.append(image1.reshape(-1,image_size,image_size,3))
emb_array1 = np.zeros((1, embedding_size))
emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]

image2 = scipy.misc.imread(image_name2, mode='RGB')
image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image2 = facenet.prewhiten(image2)
scaled_reshape.append(image2.reshape(-1,image_size,image_size,3))
emb_array2 = np.zeros((1, embedding_size))
emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]

dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
print("128维特征向量的欧氏距离：%f "%dist)