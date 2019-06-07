import tensorflow as tf

def logits(inputs):
  conv1_w = tf.get_variable('conv1_w', shape=[5, 5, 3, 64])
  conv1_b = tf.get_variable('conv1_b', shape=[64], initializer=tf.constant_initializer(0.0))
  conv1 = tf.nn.conv2d(inputs, conv1_w, strides=[1, 1, 1, 1], padding='SAME')
  conv1 = tf.nn.bias_add(conv1, conv1_b)
  conv1 = tf.nn.relu(conv1, name='conv1')
  print(conv1)

  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  print(pool1)

  conv2_w = tf.get_variable('conv2_w', shape=[5, 5, 64, 64])
  conv2_b = tf.get_variable('conv2_b', shape=[64], initializer=tf.constant_initializer(0.1))
  conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME')
  conv2 = tf.nn.bias_add(conv2, conv2_b)
  conv2 = tf.nn.relu(conv2, name='conv2')
  print(conv2)

  pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
  print(pool2)

  dim = 8 * 8 * 64
  reshape = tf.reshape(pool2, [-1, dim])
  fc3_w = tf.get_variable('fc3_w', shape=[dim, 384])
  fc3_b = tf.get_variable('fc3_b', shape=[384], initializer=tf.constant_initializer(0.1))
  fc3 = tf.nn.relu(tf.add(tf.matmul(reshape, fc3_w), fc3_b), name='fc3')
  print(fc3)

  fc4_w = tf.get_variable('fc4_w', shape=[384, 192])
  fc4_b = tf.get_variable('fc4_b', shape=[192], initializer=tf.constant_initializer(0.1))
  fc4 = tf.nn.relu(tf.add(tf.matmul(fc3, fc4_w), fc4_b), name='fc4')
  print(fc4)

  linear_w = tf.get_variable('linear_w', shape=[192, 10])
  linear_b = tf.get_variable('linear_b', shape=[10], initializer=tf.constant_initializer(0.0))
  linear = tf.add(tf.matmul(fc4, linear_w), linear_b, name='linear')
  print(linear)

  return linear

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

  reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                       if '_b' not in v.name]) * 0.001

  return cross_entropy_mean + reg_loss

def train(loss):
  opt = tf.train.GradientDescentOptimizer(0.0005)
  return opt.minimize(loss)

def accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
  score = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int32))
  return score
