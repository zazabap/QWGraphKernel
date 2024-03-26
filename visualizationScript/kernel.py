################################################    
# Author: SHiwen An.                           #
# Date: 2024-03-25.                            #
# Purpose: Kernels for svm                     #
################################################

from pl import *
from sklearn.datasets import make_classification

x,y = make_classification(n_samples = 1000)

def linear_kernel(x_i, x_j):
    return x_i.dot(x_j.T)


# self defined kenerl 
def QJSK(rho,sigma):
    s = von_neumann_entropy((rho+sigma)/2)
    a = von_neumann_entropy(rho)/2
    b = von_neumann_entropy(sigma)/2
    return s-a-b

# example for running linear kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf1 = SVC(kernel = linear_kernel)
clf1.fit(x,y)
print(f'Accuracy on Custom Kernel: {accuracy_score(y, clf1.predict(x))}')
clf2 = SVC(kernel = 'linear')
clf2.fit(x,y)
print(f'Accuracy on Inbuilt Kernel: {accuracy_score(y, clf2.predict(x))}')