import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Accessing the datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
#creating a daraframe for sex represented by bool
df_train["male"] = df_train["Sex"]=="male"
df_test["male"] = df_test["Sex"]=="male"

X = df_train[["Pclass", "male","SibSp","Fare"]].values
y = df_train["Survived"].values
X_eval = df_test[["Pclass","male","Parch","SibSp"]].values
print(X_eval)
print(X)
#splitting the training dataset into train andbtest and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=5)
#training the model 
model = LogisticRegression()
model.fit(X_train,y_train)


#predicting with the test data to avoid overfit
y_pred = model.predict(X_test)
print(y_pred)
#Scoring the model(the model scores 0.8)
print(accuracy_score(y_test, y_pred))

#Now testing the model on a new but similar data(from kaggle)
prediction = model.predict(X_eval)

print(prediction)
test_ids = df_test["PassengerId"].values

df = pd.DataFrame({"PassengerId": test_ids,"Survived": prediction})
df.to_csv("Submission.csv", index= False)