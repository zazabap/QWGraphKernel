
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x,y = make_classification(n_samples = 1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y)

# Method 1
def linear_kernel(x_i, x_j):
    return x_i.dot(x_j.T)

clf1 = SVC(kernel = linear_kernel)
clf1.fit(x_train,y_train)
print(f'Accuracy on Custom Kernel: {accuracy_score(y_test, clf1.predict(x_test))}')

clf2 = SVC(kernel = 'linear')
clf2.fit(x_train, y_train)
print(f'Accuracy on Inbuilt Kernel: {accuracy_score(y_test, clf2.predict(x_test))}')


# Method 2
def get_gram(x1, x2, kernel):
    return np.array([[kernel(_x1, _x2) for _x2 in x2] for _x1 in x1])

def RBF(x1, x2, gamma  = 1):
    return np.exp(-gamma * np.linalg.norm(x1-x2))

clf1 = SVC(kernel = 'precomputed')
print(get_gram(x_train, x_train, RBF))
clf1.fit(get_gram(x_train, x_train, RBF), y_train)
print(f'Accuracy on Custom Kernel: {accuracy_score(y_test, clf1.predict(get_gram(x_test, x_train, RBF)))}')

clf2 = SVC(kernel = 'rbf')
clf2.fit(x_train,y_train)
print(f'Accuracy on Inbuilt Kernel: {accuracy_score(y_test, clf2.predict(x_test))}')

