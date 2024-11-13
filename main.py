import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
data = pd.read_csv("real_estate_data.csv")
# print(data.head())
print(data.shape)
print(data.isna().sum())
# isna() creates a data frame of the same shape where each cell is true if it has a missing value .sum() will sum true ( missing values) as
# count of 1 so the labels have sum 0 have no missing values
data.dropna(inplace=True)
# dropna() removes the missing values from the data set inplace=True modiy the original data frame without needing to reassing it
print(data.isna().sum())
# now we have no missing values
x = data.drop(columns="MEDV")
# .drop() willtremove the specified data from the data, coulmns="MEDV" makes sure that this column is removed from the dataset 
# the remaining dataset without MEDV is assinged to x
y =data["MEDV"]
print(x.head)
print(y.head)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
regression_tree = DecisionTreeRegressor(criterion='mse')
regression_tree.fit(x_train,y_train)
