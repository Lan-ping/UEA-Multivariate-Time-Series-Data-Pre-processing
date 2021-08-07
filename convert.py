
import datasets as uea_ucr_datasets
import numpy
import math
from utils import numpy_fill
import os
uea_ucr_datasets.UEA_UCR_DATA_DIR = 'Multivariate_ts/'
dataset_names = uea_ucr_datasets.list_datasets()

def load_UEA_dataset(dataset_name):
    train_dataset = uea_ucr_datasets.Dataset(dataset_name, train=True)
    test_dataset = uea_ucr_datasets.Dataset(dataset_name, train=False)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    input_dim = train_dataset[0][0].shape[1]
    time_steps_train = max([data[0].shape[0] for data in train_dataset])
    time_steps_test = max([data[0].shape[0] for data in test_dataset])
    time_steps = max(time_steps_train, time_steps_test)

    print(train_size, test_size, time_steps, input_dim)

    train = numpy.empty((0, input_dim, time_steps))
    test = numpy.empty((0, input_dim, time_steps))
    train_labels = numpy.empty(train_size, dtype=numpy.int)
    test_labels = numpy.empty(test_size, dtype=numpy.int)

    length = []

    for i in range(train_size):
        x,y = train_dataset[i]
        x = x.transpose(1,0)
        if numpy.isnan(x).any():
            x = numpy_fill(x)
        if x.shape[1] < time_steps:
            x = numpy.pad(x, ((0, 0), (0, time_steps - x.shape[1])), 'edge')
        train = numpy.concatenate((train,numpy.expand_dims(x, axis=0)),axis=0)
        train_labels[i] = y

    for i in range(test_size):
        x,y = test_dataset[i]
        x = x.transpose(1,0)
        if numpy.isnan(x).any():
            x = numpy_fill(x)
        if x.shape[1] < time_steps:
            x = numpy.pad(x, ((0, 0), (0, time_steps - x.shape[1])), 'edge')
        test = numpy.concatenate((test,numpy.expand_dims(x, axis=0)),axis=0)
        test_labels[i] = y

    print(train.shape, test.shape)

    for j in range(input_dim):
        mean = numpy.mean(numpy.concatenate([train[:, j], test[:, j]]))
        var = numpy.var(numpy.concatenate([train[:, j], test[:, j]]))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels

def save_numpy(path, train, train_labels, test, test_labels):
    numpy.save(path+'X_train.npy',train)
    numpy.save(path+'y_train.npy',train_labels)
    numpy.save(path+'X_test.npy',test)
    numpy.save(path+'y_test.npy',test_labels)

def main():
    dataset_names = uea_ucr_datasets.list_datasets()
    all_ready = []
    for i in os.listdir('data'):
        path2 = os.path.join('data',i)  #拼接绝对路径
        if os.path.isdir(path2):      #判断如果是文件夹,调用本身
            all_ready.append(i)

    for dataset_name in list(set(dataset_names)-set(all_ready)):
        print(dataset_name)
        if not os.path.exists('data/'+dataset_name+'/'):
            os.makedirs('data/'+dataset_name+'/')
        train, train_labels, test, test_labels = load_UEA_dataset(dataset_name)
        save_numpy('data/'+dataset_name+'/', train, train_labels, test, test_labels)

     
if __name__ == '__main__':
    main()
