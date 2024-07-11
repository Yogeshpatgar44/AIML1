import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = pd.read_csv('naivetext.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
X = msg.message
y = msg.labelnum

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

count_vect = CountVectorizer()
Xtrain_dtm = count_vect.fit_transform(Xtrain)
Xtest_dtm = count_vect.transform(Xtest)

clf = MultinomialNB().fit(Xtrain_dtm, ytrain)
predicted = clf.predict(Xtest_dtm)

accuracy = metrics.accuracy_score(ytest, predicted)
conf_matrix = metrics.confusion_matrix(ytest, predicted)
precision = metrics.precision_score(ytest, predicted)
recall = metrics.recall_score(ytest, predicted)

print("Accuracy of Classifier:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
