import pandas as pd
import numpy as np
from sklearn import tree

mat_train=pd.read_csv("train.csv")
mat_train=mat_train.as_matrix()
label_train=mat_train[:,0]
features_train=mat_train[0:,1:]
print(np.shape(label_train))
print(np.shape(features_train))

mat_test=pd.read_csv("test.csv")
mat_test=mat_test.as_matrix()
features_test=mat_test
print("testing data shape...")
print(np.shape(features_test))

clf=tree.DecisionTreeClassifier()
print("Training Algo.....")
clf.fit(features_train,label_train)

print("Predicting sample....")
temp=clf.predict(features_test)
df=pd.DataFrame(data=temp.astype(float))
df.to_csv("output_decisionTree.csv",sep=',',header=False,float_format='%.2f',index=False)
print(temp)

