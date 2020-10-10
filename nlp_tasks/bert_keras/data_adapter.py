# -*- coding: utf-8 -*-


from keras.preprocessing import sequence
import numpy as np
from keras_bert import Tokenizer


def convert_data(data, maxlen, tokenizer, shuffle=False):
    """
    :param data:
    :param maxlen:
    :param shuffle:
    :param tokenizer:
    :return:
    """
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)
    X1, X2, Y = [], [], []
    for i in idxs:
        d = data[i]
        text = d[0][:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        y = d[1]
        X1.append(x1)
        X2.append(x2)
        Y.append([y])
    X1 = sequence.pad_sequences(X1, maxlen=maxlen)
    X2 = sequence.pad_sequences(X2, maxlen=maxlen)
    Y = np.array(Y)
    return [X1, X2], Y


class SingleSentenceGenerator:
    """
    """
    def __init__(self, data, tokenizer, batch_size=32, maxlen=512, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = sequence.pad_sequences(X1, maxlen=self.maxlen)
                    X2 = sequence.pad_sequences(X2, maxlen=self.maxlen)
                    Y = np.array(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


class SentencePairGenerator:
    """
    """
    def __init__(self, data, tokenizer, batch_size=32, maxlen=512, shuffle=False, class_num=2):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.class_num = class_num

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0]
                text2 = d[1]
                x1, x2 = self.tokenizer.encode(first=text1, second=text2, max_len=self.maxlen)
                if self.class_num > 2:
                    y = [0 for _ in range(self.class_num)]
                    y[d[2]] = 1
                else:
                    y = [d[2]]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = sequence.pad_sequences(X1, maxlen=self.maxlen)
                    X2 = sequence.pad_sequences(X2, maxlen=self.maxlen)
                    Y = np.array(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []
