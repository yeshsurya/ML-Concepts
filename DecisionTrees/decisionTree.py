import pydotplus
from  sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

f=open('decision_tree_data.txt','r')
x_train=[]
y_train=[]

for line in f:
    line=np.asarray(line.split(),dtype=np.float32)
    x_train.append(line[:-1])
    y_train.append(line[-1])

x_train = np.asmatrix(x_train)
y_train = np.reshape(y_train,(len(y_train),1))
clf = clf.fit(x_train,y_train)
dot_data=tree.export_graphviz(clf,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("outputtree.pdf")
print("The predection output ")
print(clf.predict([[0.,0.,1.,1.]]))
#print(x_train)
#print("----------------+++++++++++--------------------")
#print(y_train)
#clf = DecisionTreeClassifier()

#print ("Hello World")