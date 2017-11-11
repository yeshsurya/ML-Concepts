import sklearn
import numpy
from sklearn import svm
f= open('svm_data','r')

X=[]
Y=[]
for line in f:
    comps=line.strip().split('\t')
    x=[float(comps[0]),float(comps[1])]
    y=float(comps[2])
    X.append(x)
    Y.append(y)

X=numpy.asarray(X)
Y=numpy.asarray(Y)
clf = svm.SVC()
clf.fit(X, Y)
print("RBF-")
print(len(clf.support_vectors_))

linearSVC= svm.SVC(kernel='linear')
linearSVC.fit(X,Y)
print("Linear -")
print(len(linearSVC.support_vectors_))

polySVC= svm.SVC(kernel='poly')
polySVC.fit(X,Y)
print("Polynomial -")
print(len(polySVC.support_vectors_))

print("Linear Kernel support vectors : ")
print(linearSVC.support_vectors_)
print("--------------------------------")
print("Polynomial Kernel support vectors")
print(polySVC.support_vectors_)
#print(clf.predict([[2., 2.]]))


