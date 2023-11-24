import string
import csv
import torch
from torch.utils.data import Dataset


class PasswordDataset(Dataset):
    def __init__(self, data_path, alphabet, max_length=30,):
        """Create password detector data object.

        Arguments:
            data_path: The path of data in csv
            max_length: max length of a sequence
        """
        self.data_path = data_path
        self.max_length = max_length
        self.alphabet = alphabet
        self.load(data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.onehot_encode(index)
        y = self.y[index]
        return X, y

    def load(self, data_path):
        self.data = []
        self.label = []

        with open(data_path, 'r') as f:
            rdr = csv.reader(line.replace('\0', '') for line in f)
            for index, row in enumerate(rdr):
                if self.check_bad_strings(row[0]):
                    continue
                if len(row[0]) > self.max_length:
                    row[0] = row[0][0:self.max_length]
                self.data.append(row[0])
                self.label.append(int(row[1]))
        self.y = torch.LongTensor(self.label)


    def check_bad_strings(self, s):
        printable = string.printable
        for i in s:
            if i not in printable:
                return True
        return False

    def onehot_encode(self, index):
        sequence = self.data[index]
        X = torch.zeros(len(self.alphabet), self.max_length)
        for index_char, char in enumerate(sequence[::-1]):
            if self.char_to_index(char) != -1:
                X[self.char_to_index(char)][index_char] = 1.0
        return X

    def char_to_index(self, character):
        return self.alphabet.find(character)

    def get_class_weight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples / float(self.label.count(c)) for c in label_set]
        return class_weight, num_class



