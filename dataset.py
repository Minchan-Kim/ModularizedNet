import tensorflow as tf
import os


_num_data = 1
_num_time_window = 1

@tf.function
def parse(example_proto):
    feature_description = {
        'X': tf.io.FixedLenFeature(shape = (_num_data,), dtype = tf.float32),
        'y': tf.io.FixedLenFeature(shape = (2,), dtype = tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

@tf.function
def process(filename):
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse).prefetch(1)
    x_dataset = dataset.map(lambda data: data['X'])
    y_dataset = dataset.map(lambda data: data['y'])
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    return dataset

def OrderedDataset(filenames, num_data, batch_size):
    global _num_data
    _num_data = num_data

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(process)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

def Dataset(path, num_data, buffer_size, batch_size, pattern = '*.tfrecord', cycle_length = 1, num_parallel_calls = None, drop_remainder = True):
    global _num_data
    _num_data = num_data

    if num_parallel_calls is None:
        num_parallel_calls = os.cpu_count()
    cycle_length = max([cycle_length, num_parallel_calls])

    dataset = tf.data.Dataset.list_files(file_pattern = (path + '/' + pattern))
    dataset = dataset.interleave(process, cycle_length = cycle_length, num_parallel_calls = num_parallel_calls, deterministic = False)
    if buffer_size > 0:
        dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)
    dataset = dataset.prefetch(1)

    return dataset
    
    
if __name__ == "__main__":
    dataset = Dataset("/home/dyros/mc_ws/data", 65, 20000, 1000, num_parallel_calls = 1, drop_remainder = False)
    cnt = 0
    for x, y in dataset:
        cnt += (x.shape[0])
    print(cnt)  