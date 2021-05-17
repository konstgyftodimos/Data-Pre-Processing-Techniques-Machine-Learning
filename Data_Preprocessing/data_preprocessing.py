
# Libraries
import pandas as pd 
import numpy as np 
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Importing Data
df = pd.read_csv("dataset/bank.csv",sep=";")
pd.set_option("display.max_columns", 17)
print("Features are: ",df.columns)

# 2. Data Pre-Processing

# y Feature (yes/no) - Output Feature
df["y"]=df["y"].map({"no":0, "yes":1}.get)


# Age Feature (scalars)
print("Number of missing values in Age feature: ",df["age"].isna().sum())
print(" ")

# Job Feature ('admin.' 'blue-collar' 'entrepreneur' 'housemaid' 'management' 'retired' 'self-employed' 'services' 'student' 'technician' 'unemployed' 'unknown')
print("Categorical values of job feature: ",np.unique(df['job']))
df["job"]=df["job"].map({'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6, 'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 0, "unknown": np.nan}.get)
print("Number of missing values in job feature: ",df["job"].isna().sum())
df = df[df['job'].notna()]
print("Number of missing values in job feature: ",df["job"].isna().sum())
print(" ")

# Marital Feature (married/single)
print("Categorical values of marital feature: ",np.unique(df['marital']))
df["marital"]=df["marital"].map({'divorced':2, "married":1, "single":0}.get)
print("Number of missing values in marital feature: ",df["marital"].isna().sum())
print(" ")

# Education Feature ('primary' 'secondary' 'tertiary' 'unknown')
print("Categorical values of Education feature: ",np.unique(df['education']))
df["education"]=df["education"].map({'primary':0,'secondary':1, 'tertiary':2, 'unknown':np.nan}.get)
print("Number of missing values in education feature: ",df["education"].isna().sum())
df = df[df['education'].notna()]
print("Number of missing values in education feature: ",df["education"].isna().sum())
print(" ")

# Default Feature (no/yes)
print("Categorical values of default feature: ",np.unique(df['default']))
df["default"]=df["default"].map({"no":1, "yes":0}.get)
print("Number of missing values in default feature: ",df["education"].isna().sum())
print(" ")

# Balance Feature (scalar numbers, i apply minmax normalization to put everything between [0,1])
print("Number of missing values in balance feature: ",df["balance"].isna().sum())
scaler = MinMaxScaler()
df["balance"] = scaler.fit_transform(df["balance"].values.reshape(-1, 1))
print(" ")

# Housing Feature (no/yes)
print("Categorical values of Housing feature: ",np.unique(df['housing']))
df["housing"]=df["housing"].map({"no":1, "yes":0}.get)
print("Number of missing values in Housing feature: ",df["housing"].isna().sum())
print(" ")

# loan Feature (no/yes)
print("Categorical values of loan feature: ",np.unique(df['loan']))
df["loan"]=df["loan"].map({"no":1, "yes":0}.get)
print("Number of missing values in loan feature: ",df["loan"].isna().sum())
print(" ")


# Contact Feature ('cellular' 'telephone' 'unknown'), instead of deleting rows with 'nan', i replace the nan-values with the most appeared value.
print("Categorical values of contact feature: ",np.unique(df['contact']))
df["contact"]=df["contact"].map({'cellular':0, 'telephone':1 ,'unknown':np.nan}.get)
print("Number of missing values in contact feature: ",df["contact"].isna().sum())
df['contact'].fillna(value=df.mode()['contact'][0], inplace = True)
print("Number of missing values in contact feature: ",df["contact"].isna().sum())
print(" ")

# Day Feature (1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
print("Categorical values of day feature: ",np.unique(df['day']))
print("Number of missing values in day feature: ",df["day"].isna().sum())
print(" ")

# Month Feature ('apr' 'aug' 'dec' 'feb' 'jan' 'jul' 'jun' 'mar' 'may' 'nov' 'oct' 'sep')
print("Categorical values of month feature: ",np.unique(df['month']))
df["month"]=df["month"].map({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}.get)
print("Number of missing values in month feature: ",df["month"].isna().sum())
print(" ")

# Duration Feature (scalar numbers, i apply minmax normalization to put everything between [0,1])
print("Number of missing values in duration feature: ",df["duration"].isna().sum())
scaler = MinMaxScaler()
df["duration"] = scaler.fit_transform(df["duration"].values.reshape(-1, 1))
print(" ")

# Campaign Feature (scalar, leave as it is, only check for nan values)
print("Number of missing values in campaign feature: ",df["campaign"].isna().sum())
print(" ")

# Pdays Feature (scalar, i apply minmax normalization to put everything between [0,1])
print("Number of missing values in pdays feature: ",df["pdays"].isna().sum())
scaler = MinMaxScaler()
df["pdays"] = scaler.fit_transform(df["pdays"].values.reshape(-1, 1))
print(" ")

# Previous Feature (scalar, leave as it is, only check for nan values)
print("Number of missing values in previous feature: ",df["previous"].isna().sum())
print(" ")

# Poutcome Feature ('failure' 'other' 'success' 'unknown') # IT HAS A LOT OF NAN values
print("Categorical values of poutcome feature: ",np.unique(df['poutcome']))
df["poutcome"]=df["poutcome"].map({'failure':0, 'other':2, 'success':1, 'unknown': np.nan}.get)
print("Number of missing values in poutcome feature: ",df["poutcome"].isna().sum())
print(" ")

# Poutcome Feature, use linear regression to predict NaN values.
# Swap 'y' with 'poutcome' and create new dataframe
columns_titles = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing','loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'y', 'poutcome']
df_2=df.reindex(columns=columns_titles)
# Seperate the NULL values from "Poutcome"-feature
test_data = df_2[df_2["poutcome"].isnull()]
# X_train: Drop the NULL values from my dataframe 
df_2.dropna(inplace=True)
X_train = df_2.drop("poutcome", axis = 1)
# y_train: Keep rows from the df_2["poutcome"], with non-Nan values:
y_train = df_2["poutcome"]
# X_test: create the X_test from the test data
X_test = test_data.drop("poutcome", axis = 1 )
# Apply logistic regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
# Replace missing values with predicted values in the initial dataframe:
df.loc[df.poutcome.isnull(), 'poutcome'] = y_pred

# Find Correlations Between Features and Final Response - 'y'
df.corr()
import matplotlib.pyplot as plt
plt.matshow(df.corr(), cmap = "summer")
plt.xticks(list(range(len(df.columns))),df.columns, rotation='vertical')
plt.yticks(list(range(len(df.columns))),df.columns, rotation='horizontal')
plt.show()
print(df.corr()["y"].sort_values(ascending = False))

# Apply PCA to highly correlated features to reduce dimensionality:
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 1. pdays/housing/previous - features have similar negative correlation, so reduce them to 1 feature, import them to df and delete the three reduced features.
x_1 = df.loc[:,['pdays','housing','previous']]
x_1 = StandardScaler().fit_transform(x_1)
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x_1)
principalDf_1 = pd.DataFrame(data = principalComponents, columns = ['Feature1'])
# follow similar procedure for other feature if needed :)



# Detect outliers with boxplot diagram:
df.plot.box()
plt.xticks(list(range(len(df.columns))),df.columns, rotation='vertical')
plt.show()
# Outliers in categorical data are not a problem, so we only need to remove the outliers from feature - age:
low = 0.01
high = 0.99
qdf = df.quantile([low,high])
df.age = df.age.apply(lambda v: v if qdf.age[low] < v < qdf.age[high] else np.nan)
df.pdays = df.pdays.apply(lambda v: v if qdf.pdays[low] < v < qdf.pdays[high] else np.nan)

# See outliers with boxplot diagram again:
df.plot.box()
plt.xticks(list(range(len(df.columns))),df.columns, rotation='vertical')
plt.show()







