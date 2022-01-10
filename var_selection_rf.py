# Jose Caloca jc110558

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# load dataset
df = pd.read_csv('./dataset_project.csv')

df = df[df.country == 'Poland']

df.reset_index(drop=True, inplace=True)


# convert "-9" into NaN

for col in df.columns:
    df[col] = df[col].replace(to_replace=-9, value=np.nan)
    
# set target variable

y = df.d2
y
# filter variables with <30% of missing data

filter = df.isnull().sum()/len(df) < 0.3
df = df.loc[: , filter]
df = pd.concat([y, df], axis=1)
# label encoding (note: this is different than one-hot encoding)

numerical = df.select_dtypes(exclude=['object'])
categorical = df.select_dtypes(include=['object'])

list = []
for i in categorical.columns:
    df[i] = df[i].astype('category')
    df[i] = df[i].cat.codes
    list.append(df[i])
    

columns = categorical.columns
categorical = pd.DataFrame(np.array(list).transpose(), columns=columns)


df = pd.concat([numerical, categorical], axis=1)
sorted = df.isnull().sum()/len(df)
sorted.sort_values()

# Apply multiple imputation to the values

imp_mean = IterativeImputer(random_state=0)
imp_mean = imp_mean.fit(df[df.columns])
df[df.columns] = imp_mean.transform(df[df.columns])

sorted = df.isnull().sum()/len(df)
sorted.sort_values()

# target and features selection

X = df.loc[ : , df.columns != 'd2']
y = df['d2']

# split dataset
np.random.seed(seed = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)


# train random forest
rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)

rf.fit(X_train, y_train)

# get variable importance
var_importance = dict(variables = X_train.columns, importance = rf.feature_importances_)
var_importance = pd.DataFrame(var_importance)
var_importance.sort_values(by=['importance'], inplace=True, ascending=False)

# bar plot of importance of all variables
plt.bar(var_importance['variables'], var_importance['importance'])
plt.show()

# variables that affect to "d2" by more than 2%
top_variables = var_importance.query('importance > 0.02')
plt.bar(top_variables['variables'], top_variables['importance'])
plt.show()