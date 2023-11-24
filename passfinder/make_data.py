import csv
from json.tool import main
import random

random.seed(2021)
language_list = ["py", "go", "js", "ts", "java", "c", "cpp", "cs", "php", "rb"]


def read_raw_csv(csvfile):
    pos, neg = [], []
    with open(csvfile, "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            context, label = row[0], row[1]
            if label == '1':
                pos.append(row[0])
            else:
                neg.append(row[0])
    random.shuffle(pos)
    random.shuffle(neg)
    return pos, neg


def make_context_by_language(csvfile):
    all = {}
    for i in language_list:
        all[i] = []
    with open(csvfile, "r") as f:
        rdr = csv.reader(f)
        for row in rdr:
            context, label, language = row[0], row[1], row[2]
            all[language].append((context, label))

    for i in language_list:
        train_file = "context_" + i + "_train.csv"
        test_file = "context_" + i + "_test.csv"

        with open(train_file, "w") as trainfile:
            with open(test_file, "w") as testfile:
                train_writer = csv.writer(trainfile)
                test_writer = csv.writer(testfile)
                for language in all:
                    if language == i:
                        for context, value in all[language]:
                            test_writer.writerow([context, value, language])
                    else:
                        for context, value in all[language]:
                            train_writer.writerow([context, value, language])


def make_context_n_fold(csvfile, fold=10):
    pos, neg = read_raw_csv(csvfile)
    pos_fold_num = len(pos) // fold
    neg_fold_num = len(neg) // fold

    pos_folds = []
    neg_folds = []

    for i in range(fold):
        pos_folds.append(pos[i * pos_fold_num: (i + 1) * pos_fold_num])
        neg_folds.append(neg[i * neg_fold_num: (i + 1) * neg_fold_num])

    for i in range(fold):
        train_file = "fold_" + str(i) + "_train.csv"
        test_file = "fold_" + str(i) + "_test.csv"
        with open(train_file, "w") as trainfile:
            with open(test_file, "w") as testfile:
                train_writer = csv.writer(trainfile)
                test_writer = csv.writer(testfile)
                for j in range(fold):
                    if j == i:
                        for context in neg_folds[j]:
                            test_writer.writerow([context, 0])
                        for context in pos_folds[j]:
                            test_writer.writerow([context, 1])
                    else:
                        for context in neg_folds[j]:
                            train_writer.writerow([context, 0])
                        for context in pos_folds[j]:
                            train_writer.writerow([context, 1])

if __name__ == '__main__':
    make_context_n_fold("../data/context.csv")