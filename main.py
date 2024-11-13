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
# .drop() willl extract data from the data coulmns="MEDV" makes sure that this column is removed from the dataset and assigned to x
