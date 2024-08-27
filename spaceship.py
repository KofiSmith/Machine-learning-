import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


#Accessing data from  the train and test files
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


#Converting HomePlanet features to numerical values in train data and storing it in a data frame
ordinal_encoder = OrdinalEncoder()
Home = df_train[["HomePlanet"]]
HomeP= ordinal_encoder.fit_transform(Home)
df_train["HomeP"] = HomeP
#Converting HomePlanet ffeatures to numerical values
#Converting HomePlanet features to numerical values in test data and storing it in a data frame
ordinal_encoder = OrdinalEncoder()
Home = df_test[["HomePlanet"]]
HomeP= ordinal_encoder.fit_transform(Home)
df_test["HomeP"] = HomeP


#Accessing relevant features from the datas
X_train = df_train[["RoomService","FoodCourt","ShoppingMall","Spa"]].values
X_eval = df_test[["RoomService","FoodCourt","ShoppingMall","Spa"]].values
y = df_train["Transported"].values
#Splitting data into train and test datas
X_train, X_test, y_train, y_test = train_test_split(X_train,y, random_state = 101)

#Feeding train datas into the ML model
model = DecisionTreeClassifier(random_state=101)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred[:20])
print(y_test[:20])
print(accuracy_score(y_test, y_pred))

#Testing the model on the evaluation data
prediction = model.predict(X_eval)


#Saving predictions from the evaluation data into csv file
test_ids = df_test["PassengerId"].values

df = pd.DataFrame({"PassengerId": test_ids,"Transported": prediction})
df.to_csv("spaceship.csv", index= False)