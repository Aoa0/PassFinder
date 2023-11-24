import torch
import string
import csv
from tqdm import tqdm


def char_to_index(character, alphabet):
    return alphabet.find(character)


def one_hot_encode(seq, alphabet, max_length=30):
    X = torch.zeros(len(alphabet), max_length)
    for index_char, char in enumerate(seq[::-1]):
        if char_to_index(char, alphabet) != -1:
            X[char_to_index(char, alphabet)][index_char] = 1.0
    return X


def predict(text, model, max_length, alphabet, cuda=False):
    assert isinstance(text, str)
    with torch.no_grad():
        model.eval()
        x = one_hot_encode(text, alphabet, max_length)
        x = torch.unsqueeze(x, 0)
        if cuda:
            x = x.cuda()
        output = model(x)
        # print(torch.exp(output))
        _, predicted = torch.max(output, 1)
        return predicted, torch.exp(output)


def predict_finetuned(text, model, max_length, alphabet, cuda=False):
    assert isinstance(text, str)
    with torch.no_grad():
        model.eval()
        x = one_hot_encode(text, alphabet, max_length)
        x = torch.unsqueeze(x, 0)
        if cuda:
            x = x.cuda()
        output = model(x)
        prob = torch.exp(output)[0]
        if prob[0] > 0.2:
            pred = 0
        else:
            if prob[1] > prob[2]:
                pred = 1
            else:
                pred = 2
        return pred, prob
