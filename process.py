import numpy as np
import pandas as pd
import librosa


def __parser(row):
    '''
    根据DataFrame的每一行处理数据
    :param row: (filename,label)
    :return:
    '''

    filename = row.filename

    try:
        data, sample_rate = librosa.load(filename, mono=True, res_type='kaiser_fast', sr=None)
        data = __get_start_point(data, 2000)
    except Exception as e:
        print("Error encountered while parsing file: %s \n" % filename)
        return None, None

    label = (row.label if row.label < 5 else 0)

    return [data, label]


def __get_start_point(data, split_time):
    processed_data = np.zeros(split_time)
    i = 0
    while -0.1 < data[i] < 0.1:
        i = i + 1
    else:
        if i + split_time <= len(data):
            processed_data = data[i:i + split_time]
        else:
            processed_data[:len(data) - i] = data[i:]
    return processed_data


def preprocessing(filename):
    '''
    数据预处理
    :return:
    '''
    filename_label = pd.read_csv(filename, header=None)
    filename_label.columns = ['filename', 'label']
    data_label = filename_label.apply(__parser, axis=1)
    data_label.columns = ['feature', 'label']

    from sklearn.preprocessing import LabelEncoder
    import keras.utils.np_utils as np_utils

    wav_data = np.array(data_label.feature.tolist())
    label = np.array(data_label.label.tolist())

    lb = LabelEncoder()
    label = np_utils.to_categorical(lb.fit_transform(label))
    return wav_data, label


def batch_data(source, target, batch_size):
    for batch_i in range(0, len(source) // batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch


if __name__ == '__main__':
    wav_data, label = preprocessing('label.csv')
    print(wav_data)
    print(label)
