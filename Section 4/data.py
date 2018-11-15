# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for downloading and reading MNIST data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import random

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
# from tensorflow.python.util.deprecation import deprecated

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return dense_to_one_hot(labels, num_classes)


class DataSet(object):

  def __init__(self,
               images,
               labels):
    # reshaping
    images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    
    #shuffle
    perm0 = numpy.arange(self._num_examples)
    numpy.random.shuffle(perm0)
    self._images = self.images[perm0]
    self._labels = self.labels[perm0]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    rand = random.randint(0, self._num_examples - batch_size)
    start = rand
    end = rand + batch_size
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   validation_size=5000):

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  with gfile.Open(train_dir + "/" + TRAIN_IMAGES, 'rb') as f:
    train_images = extract_images(f)

  with gfile.Open(train_dir + "/" + TRAIN_LABELS, 'rb') as f:
    train_labels = extract_labels(f)

  with gfile.Open(train_dir + "/" + TEST_IMAGES, 'rb') as f:
    test_images = extract_images(f)

  with gfile.Open(train_dir + "/" + TEST_LABELS, 'rb') as f:
    test_labels = extract_labels(f)

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)

