import tensorflow as tf
import numpy as np
from collections import deque
from pathlib import Path
import argparse


max_data = []

min_data = []

variables_dict = {
    'time': [0, 1], 
    'otee': [1, 17], 
    'tauj': [17, 24], 
    'dtauj': [24, 31], 
    'q': [31, 38], 
    'qd': [38, 45], 
    'dq': [45, 52], 
    'dqd': [52, 59], 
    'ddqd': [59, 66], 
    'tauext': [66, 72], 
    'ofext': [72, 79], 
    'theta': [79, 86], 
    'dtheta': [86, 93], 
    'tauc': [93, 100], 
    'taudyn': [100, 107],
    'mass': [107, 156],
    'cor': [156, 163],
    'grav': [163, 170],
    'jointc': [170, 177],
    'cart': [177, 183]
}

def max_min(paths):
    max_values = []
    min_values = []
    for i in range(107):
        max_values.append(deque(maxlen = 20))
        min_values.append(deque(maxlen = 20))

    for path in paths:
        data = pd.read_csv(path, header = None).to_numpy()
        for i in range((data.shape)[0]):
            for j in range(107):
                if len(max_values[j]) == 0:
                    max_values[j].append(data[i, j])
                else:
                    if data[i, j] > min(max_values[j]):
                        max_values[j].append(data[i, j])
                if len(min_values[j]) == 0:
                    min_values[j].append(data[i, j])
                else:
                    if data[i, j] < max(min_values[j]):
                        min_values[j].append(data[i, j])

    for i in range(107):
        max_data.append(min(max_values[i]))
        min_data.append(max(min_values[i]))

    np.savetxt('max.csv', np.array(max_data), delimiter = ',')
    np.savetxt('min.csv', np.array(min_data), delimiter = ',')

def get_max_min():
    data_max = pd.read_csv('/home/dyros/mc_ws/ModularizedNet/training/max.csv', header = None).to_numpy()
    for i in range((data_max.shape)[0]):
        max_data.append(data_max[i])
    data_min = pd.read_csv('/home/dyros/mc_ws/ModularizedNet/training/min.csv', header = None).to_numpy()
    for i in range((data_min.shape)[0]):
        min_data.append(data_min[i])

def normalize_data(data, i, j):
    return (2 * (data[i:j] - min_data[i:j]) / (max_data[i:j] - min_data[i:j]) - 1)

def Normalize(src, dest, timesteps, indices, frequency):
    logdata = pd.read_csv(src, header = None).to_numpy()
    size = (logdata.shape)[0]
    num_data = 0
    for index in indices:
        num_data += (index[1] - index[0])
    num_data += 2
    normalized_data = np.zeros((size, num_data), dtype = logdata.dtype)
    start = 0
    for i in range(size):
        if (i > 0) and (round((logdata[i, 0] - logdata[(i - 1), 0]), 3) != (1 / float(frequency))):
            if (i - start) >= timesteps:
                write(normalized_data[start:i, :], src, dest, timesteps)
            start = i
        data = logdata[i, :]
        j = 0
        for index in indices:
            k = (j + index[1] - index[0])
            normalized_data[i, j:k] = normalize_data(data, index[0], index[1])
            j = k
        normalized_data[i, -2:] = data[-2:]
    if (size - start) >= timesteps:
        write(normalized_data[start:size, :], src, dest, timesteps)

def write(data, src, dest, timesteps):
    windows = []
    window = deque(maxlen = timesteps)
    for i in range(timesteps - 1):
        window.append(data[i, :-2])
    for i in range((timesteps - 1), (data.shape)[0]):
        window.append(data[i, :-2])
        temp = np.vstack(tuple(window))
        x = np.zeros_like(temp)
        m = 0
        for j in range(7):
            for k in timesteps:
                for l in range((data.shape)[1] / 7):
                    x[m] = temp[(l * 7 + j) + ((data.shape)[1] * k)]
                    m += 1    
        y = data[i, -2:]
        windows.append((x, y))
    np.random.shuffle(windows)
    src = str(src)
    file_index = int(src[3:-4])
    filename = dest + '/training{}'.format(file_index)
    writer = tf.io.TFRecordWriter(filename  + '.tfrecord')
    for x, y in windows:
        feature = {
            'x': tf.train.Feature(bytes_list = tf.train.FloatList(value = x)),
            'y': tf.train.Feature(float_list = tf.train.FloatList(value = y))
        }
        example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
        writer.write(example_proto.SerializeToString())
    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', nargs = '?', default = '/home/dyros/mc_ws/ModularizedNet/data/training')
    parser.add_argument('dest_path',nargs = '?', default = '/home/dyros/mc_ws/ModularizedNet/data/training')
    parser.add_argument('--dataset', type = str, default = 'training')
    parser.add_argument('--variables', nargs = '+')
    args = parser.parse_args()
    src_path = args.src_path
    dest_path = args.dest_path
    dataset = args.dataset
    variables = args.variables
    variables = [variable.lower() for variable in variables]
    variables = [variable.replace('_', '') for variable in variables]

    tf.get_logger().setLevel('WARN')

    paths = list(Path(src_path).glob('*.txt'))
    indices = []
    for key in variables_dict.keys():
        for variable in variables:
            if variable.startswith(key):
                indices.append(variables_dict[key])
                break
    indices.sort(key = (lambda index: index[0]))

    if dataset == 'training':
        max_min(paths)
    else:
        get_max_min()
    
    for path in paths:
        Normalize(path, dest, timesteps, indices, 100)
        print(str(path))
