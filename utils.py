import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import codecs
import collections
import glob
import numpy as np
import itertools
import random
from six.moves import cPickle
import time

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8', train_percentage=0.5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        self.file_number = 0
        self.num_of_files = sum(1 for item in glob.iglob(os.path.join(self.data_dir, 'input*.txt')))
        self.vocabulary_created = False

        self.num_train_files = int(self.num_of_files * train_percentage)
        self.num_test_files = self.num_of_files - self.num_train_files

        input_file, vocab_file, tensor_file = self._get_filenames(self.file_number)
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("Reading text file")
            self._preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Loading preprocessed files")
            self._load_preprocessed(vocab_file, tensor_file)
        self._create_batches()
        self._reset_batch_pointer()

    def _get_filenames(self, file_number):
        input_file = os.path.join(self.data_dir, "input{}.txt".format(file_number))
        vocab_file = os.path.join(self.data_dir, "vocab.pkl")
        tensor_file = os.path.join(self.data_dir, "data{}.npz".format(file_number))
        return input_file, vocab_file, tensor_file

    def _create_vocabulary(self, data, vocab_file):
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)

        self.vocabulary_created = True

    def _preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        # self.len_of_words = np.loadtxt(self.data_dir + "/length" + str(self.file_number) + ".txt", dtype=int) + 1

        self._create_vocabulary("abcdefghijklmnopqrstuvwxyzßäöü ", vocab_file)

        words = data.split()
        self.tensor = []
        self.word_lengths = []
        count = len(words)
        for index, word in enumerate(words):
            if len(word) < self.seq_length - 1:
                seq = np.full(self.seq_length, self.vocab[' '])
                try:
                    seq[1:len(word) + 1] = np.array(list(map(self.vocab.get, word)))
                except TypeError:
                    print(word)
                    break
                self.tensor.append(seq)
                self.word_lengths.append(len(word) + 2)
                if (index + 1) % 10000 == 0:
                    print('{}/{} words parsed'.format(index + 1, count))
        np.savez(tensor_file, tensor=self.tensor, word_lengths=self.word_lengths)

    def _load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        data = np.load(tensor_file)
        self.tensor = data['tensor']
        self.word_lengths = data['word_lengths']

    def _create_batches(self):
        self.num_batches = int(len(self.tensor) / self.batch_size)

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        dataset = self.tensor[:self.num_batches * self.batch_size]

        x = np.array(dataset)
        timer = time.clock()
        random.shuffle(x)
        print('{:.2f} seconds for shuffle'.format(time.clock() - timer))

        y = x.copy().reshape(-1)
        nil = y[0]
        y[:-1] = y[1:]
        y[-1] = nil
        y = y.reshape(x.shape)

        self.xs = np.split(x, self.num_batches)
        self.ys = np.split(y, self.num_batches)
        self.word_lengths = np.array(self.word_lengths)
        self.wls = np.split(self.word_lengths[:self.num_batches * self.batch_size], self.num_batches)

    def iterbatches(self, mode):
        while True:
            if self.pointer >= self.num_batches:
                self.file_number += 1
                if (mode == "train" and self.file_number >= self.num_train_files
                        or mode == "test" and self.file_number >= self.num_of_files):
                    return

                self._reset_batch_pointer()

                input_file, vocab_file, tensor_file = self._get_filenames(self.file_number)
                if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
                    print("Reading text file")
                    self._preprocess(input_file, vocab_file, tensor_file)
                else:
                    print("Loading preprocessed files")
                    self._load_preprocessed(vocab_file, tensor_file)
                # self._preprocess(input_file, vocab_file, tensor_file)
                self._create_batches()
            x, y, word_lengths = self.xs[self.pointer], self.ys[self.pointer], self.wls[self.pointer]
            self.pointer += 1
            yield x, y, word_lengths

    def _reset_batch_pointer(self):
        self.pointer = 0
