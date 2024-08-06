import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


cancer_data = load_breast_cancer()
print(cancer_data.keys())
#print(cancer_data["DESCR"])

df = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
print(df.head())
#print(cancer_data["target"])
#print(cancer_data["target"].shape)
#print(cancer_data["target_names"])

df["target"] = cancer_data["target"]
print(df.head())

X = df[cancer_data.feature_names].values
y = df["target"].values


#Splitting the dataset into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 27)

#Building the model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("SPLITTING THE DATA SET INTO TRAINING AND TESTING DATAS")

print("whole data set :", X.shape, y.shape)
print("training set :", X_train.shape, y_train.shape)
print("test data set :", X_test.shape, y_test.shape)


#print("prediction for datapoint 0:",  model.predict([X[0]]))
#print(model.score(X,y))

print(" ")
#Evaluating the model
#Testing accuracy, precision and recall of data set
print("Accuracy :", accuracy_score(y_test, y_pred))
print("precision :", precision_score(y_test, y_pred))
print("recall :", recall_score(y_test, y_pred))
print("f1 score :", f1_score(y_test, y_pred))

#scoring the overall model
print("model score :", model.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))

print(model.predict(X_test[:20]))
print(y_test[:20])
