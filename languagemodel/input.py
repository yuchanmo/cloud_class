import numpy as np
import tensorflow as tf

RECORD_LENGTH = 1 + 3072 # 1: label, 3072: rgb (32 * 32 * 3)


def parse_data(data_bytes):
  data = tf.decode_raw(data_bytes, tf.uint8)
  label, rgb = tf.split(data, [1, RECORD_LENGTH - 1], axis=0)
  label = tf.cast(label, tf.int32)
  label = tf.reshape(label, [])
  rgb = tf.cast(rgb, tf.float32)
  rgb = tf.reshape(rgb, [3, 32, 32])
  rgb = tf.transpose(rgb, [1, 2, 0])

  return label, rgb

def get_iterator(filenames, batch_size):
  ds = tf.data.Dataset.from_tensor_slices(filenames)
  ds = ds.apply(
      # Read and preprocess cycle_length files concurrently
      tf.contrib.data.parallel_interleave(
          lambda filename: tf.data.FixedLengthRecordDataset(filename, RECORD_LENGTH),
          cycle_length=len(filenames)))

  # Prefetch BATCH_SIZE items
  ds = ds.prefetch(batch_size)

  ds = ds.shuffle(50000)
  ds = ds.repeat()

  # Apply parse_toy_data in parallel
  ds = ds.map(parse_data, num_parallel_calls=batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(1)

  iterator = ds.make_one_shot_iterator()
  return iterator
