import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(2021)

Corpus = pd.read_csv(r"../data/context.csv", encoding='utf-8')
Corpus['text'].dropna(inplace=True)
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]

for index, entry in enumerate(Corpus['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word)
            Final_words.append(word_Final)
        else:
            pass
    Corpus.loc[index, 'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'],
                                                                    test_size=0.1)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ", accuracy_score(Test_Y, predictions_NB))
print("Naive Bayes F1 Score -> ", f1_score(Test_Y, predictions_NB))
print("Naive Bayes Precision Score -> ", precision_score(predictions_NB, Test_Y))
print("Naive Bayes Recall Score -> ", recall_score(predictions_NB, Test_Y))
print("\n")


SVM = svm.SVC(C=0.5, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(Test_Y, predictions_SVM))
print("SVM F1 Score -> ", f1_score(Test_Y, predictions_SVM))
print("SVM Precision Score -> ", precision_score(Test_Y, predictions_SVM))
print("SVM Recall Score -> ", recall_score(Test_Y, predictions_SVM))
print("\n")

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Train_X_Tfidf, Train_Y)
predictions_KNN = neigh.predict(Test_X_Tfidf)
print("Knn Accuracy Score -> ", accuracy_score(Test_Y, predictions_KNN))
print("Knn F1 Score -> ", f1_score(Test_Y, predictions_KNN))
print("Knn Precision Score -> ", precision_score(Test_Y, predictions_KNN))
print("Knn Recall Score -> ", recall_score(Test_Y, predictions_KNN))

print("\n")
clf = LogisticRegression(random_state=0)
clf.fit(Train_X_Tfidf, Train_Y)
predictions_clf = clf.predict(Test_X_Tfidf)
print("LR Accuracy Score -> ", accuracy_score(Test_Y, predictions_clf))
print("LR F1 Score -> ", f1_score(Test_Y, predictions_clf))
print("LR Precision Score -> ", precision_score(Test_Y, predictions_clf))
print("LR Recall Score -> ", recall_score(Test_Y, predictions_clf))
