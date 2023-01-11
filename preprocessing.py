import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set path for the input
RAW_DATA_DIR = os.environ["RAW_DATA_DIR"]
RAW_DATA_FILE = os.environ["RAW_DATA_FILE"]
raw_data_path = os.path.join(RAW_DATA_DIR, RAW_DATA_FILE)

# Read dataset
df = pd.read_csv(raw_data_path, sep=",")

num = df.select_dtypes(exclude='O')
numeric = num.iloc[:,:-1]

for i in numeric:
#     print(i)
    q1 = np.quantile(numeric[i],0.25)
    q3 = np.quantile(numeric[i],0.75)
    
    iqr = q3 - q1
    
    lq = (q1-1.5*iqr)
    uq = (q3+1.5*iqr)
    for j in numeric[i].values:
#         print(j)
        if j > uq:
            numeric[i] = numeric[i].replace(j,uq)
            
        if j < lq:
            numeric[i] = numeric[i].replace(j,lq)
            
df.drop(numeric.columns,axis=1,inplace=True)

df.drop('Surname',axis=1,inplace=True)

df = pd.get_dummies(df,drop_first=True)

data = pd.concat([numeric,df],axis=1)


# Split into dependend and independent variables

X = df.drop('Exited',axis=1)
y = df['Exited']


# Split into train and test
train, test = train_test_split(data, test_size=0.3, stratify=data['Exited'])


# Set path to the outputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# Save csv
train.to_csv(train_path, index=False)
test.to_csv(test_path,  index=False)