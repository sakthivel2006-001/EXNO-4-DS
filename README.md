# EXNO:4 : Feature Scaling and Selection

### NAME: SAKTHIVEL S

### REG.NO:212223220090

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

 ```

import pandas as pd
import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

```

<img width="1388" height="564" alt="image" src="https://github.com/user-attachments/assets/6cab13a8-3da2-4349-8c4a-f76208fe0f69" />

```
data.isnull().sum()

```

<img width="494" height="612" alt="image" src="https://github.com/user-attachments/assets/c2443eec-47c4-4ebd-87ab-73de02d0a1c8" />


```
missing=data[data.isnull().any(axis=1)]
missing

```
<img width="1379" height="600" alt="image" src="https://github.com/user-attachments/assets/2aeaef4f-1ef5-46e2-9f0a-1b2b7053f923" />


```

data2=data.dropna(axis=0)
data2

```

<img width="1717" height="633" alt="image" src="https://github.com/user-attachments/assets/dbf4d1d1-a732-4fdc-9bf5-317d66011b83" />


```

sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```

<img width="558" height="263" alt="image" src="https://github.com/user-attachments/assets/4f9e136f-3fa0-468a-8025-c3335549c85b" />

```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs

```

<img width="441" height="513" alt="image" src="https://github.com/user-attachments/assets/73cb1731-128b-4ec3-a4b8-e7bfd42632fb" />


```

data2

```
<img width="1584" height="542" alt="image" src="https://github.com/user-attachments/assets/c73ea92d-8657-4072-9522-9adbc2606df1" />


```

new_data=pd.get_dummies(data2, drop_first=True)
new_data

```

<img width="1758" height="600" alt="image" src="https://github.com/user-attachments/assets/f860dd94-f07c-4e81-aaef-acf14bbcdaa3" />


```

columns_list=list(new_data.columns)
print(columns_list)

```

<img width="1746" height="46" alt="image" src="https://github.com/user-attachments/assets/52442b75-8736-4744-823b-e2b9870221ce" />

```

features=list(set(columns_list)-set(['SalStat']))
print(features)

```
<img width="1725" height="47" alt="image" src="https://github.com/user-attachments/assets/883505fc-9a3b-4057-a261-2c9efab00be1" />


```

y=new_data['SalStat'].values
print(y)

```

<img width="328" height="36" alt="image" src="https://github.com/user-attachments/assets/2582eb96-be7c-43ec-bd98-5346649ff916" />

```

x=new_data[features].values
print(x)

```
<img width="776" height="169" alt="image" src="https://github.com/user-attachments/assets/7eadfa88-382c-40ba-b955-257b8be6086e" />

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```
<img width="554" height="92" alt="image" src="https://github.com/user-attachments/assets/7708d57c-6e5c-4dde-bfb5-6587e3a0c3f8" />

```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
<img width="293" height="28" alt="image" src="https://github.com/user-attachments/assets/cb3b93e6-f185-41ef-9819-aa48a1d32e4a" />

```

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```
<img width="385" height="47" alt="image" src="https://github.com/user-attachments/assets/5b9e6d77-7967-410e-88ef-ec77dec5aded" />

```

data.shape

```

<img width="203" height="26" alt="image" src="https://github.com/user-attachments/assets/fae53646-2875-43bb-a1a3-336d9690e442" />

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

<img width="571" height="53" alt="image" src="https://github.com/user-attachments/assets/da5e92a3-1f64-489d-884b-50fec0c9cee5" />

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```

<img width="576" height="240" alt="image" src="https://github.com/user-attachments/assets/4a7c6e36-ad4d-4791-b2a5-08396245fe64" />


```
tips.time.unique()

```

<img width="506" height="47" alt="image" src="https://github.com/user-attachments/assets/d214b88c-b825-4562-ab24-0072e57d1c48" />

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```

<img width="579" height="95" alt="image" src="https://github.com/user-attachments/assets/27b68156-3a4a-45ce-8099-928364c1e2ae" />


```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

```


<img width="555" height="82" alt="image" src="https://github.com/user-attachments/assets/f47d3caf-056a-4dbe-a52a-5a7783f6ec3e" />



# RESULT:
     Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.
