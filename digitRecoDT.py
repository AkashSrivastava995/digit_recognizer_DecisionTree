# importing important libraries
import pandas as pd
import numpy as np
from sklearn import tree

# reading training data from train.csv
mat_train=pd.read_csv("train.csv")
mat_train=mat_train.as_matrix()
label_train=mat_train[:,0]
features_train=mat_train[0:,1:]
print(np.shape(label_train))
print(np.shape(features_train))

# reading test data from test.csv
mat_test=pd.read_csv("test.csv")
mat_test=mat_test.as_matrix()
features_test=mat_test
print("testing data shape...")
print(np.shape(features_test))

# Creating a instance of DecisionTree Classifier
clf=tree.DecisionTreeClassifier()

# Training the Classifier on Train data
print("Training Algo.....")
clf.fit(features_train,label_train)

# Predicting unkown data (Test)
print("Predicting sample....")
temp=clf.predict(features_test)

# Saving the output to file name `output_decisionTree.csv`
df=pd.DataFrame(data=temp.astype(float))
df.to_csv("output_decisionTree.csv",sep=',',header=False,float_format='%.2f',index=False)
print(temp)

