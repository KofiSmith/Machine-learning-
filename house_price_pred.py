import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


df_train= pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train["Y"] = df_train["CentralAir"]=="Y"
df_test["Y"] = df_test["CentralAir"]=="Y"


#Converting HomePlanet features to numerical values in train data and storing it in a data frame
#ordinal_encoder = OrdinalEncoder()
#Lot = df_train[["LotShape"]]
#LotS= ordinal_encoder.fit_transform(Lot)
#df_train["LotShape"] = LotS
#Converting HomePlanet ffeatures to numerical values
#Converting HomePlanet features to numerical values in test data and storing it in a data frame
ordinal_encoder = OrdinalEncoder()
Lot = df_test[["LotShape"]]
LotS= ordinal_encoder.fit_transform(Lot)
df_test["LotShape"] = LotS


X = df_train[["OverallQual","LotArea","MSSubClass","OverallCond"]].values
y = df_train["SalePrice"].values
X_eval = df_test[["MSSubClass","OverallQual","OverallCond","Y"]].values


print(X.shape)
print(df_test["LotShape"].shape)
#print(y.shape)
#X_test = df_test[["OverallQual","OverallCond"]].values
#y_test = df_test["SalePrice"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 5)




model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_test.shape)
print(y_pred.shape)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

"""
prediction = model.predict(X_eval)


test_ids = df_test["Id"].values

df = pd.DataFrame({"Id": test_ids,"SalePrice": prediction})
df.to_csv("house_pred_submission.csv", index= False)
"""