import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

data = np.genfromtxt("colorrectal_2_classes_formatted.txt", delimiter=",")

classes = data[:, 142]
attributes = np.delete(data,(142), axis=1)
min_max_scaler = MinMaxScaler()
attributes_norm = min_max_scaler.fit_transform(attributes)

x_train, x_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.2)
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}

svm = SVC(gamma='auto')
clf = GridSearchCV(svm, parameters).fit(x_train, y_train)

means = clf.cv_results_['mean_test_score']
std = clf.cv_results_['std_test_score']

y_pred = clf.predict(x_test)

for mean, std, params in zip(means, std, clf.cv_results_['params']):
    print('mean: %0.3f std: %0.03f para %r' %(mean, std*2, params))
    print('----------------------------------------------------------')
print('Os melhores parâmetros encontrados foram: \n', clf.best_params_)

prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("acc: ", acc)
print("prec: ", prec)
print("f1: ", f1)
print("recall: ", recall)