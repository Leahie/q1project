import pandas as pd 
from tqdm import tqdm 
from scipy import stats
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


clas = "Class"

df = pd.read_csv('Alzheimer_s_Disease_and_Healthy_Aging_Data.csv')
print(len(df))
df = df.drop(['Data_Value_Footnote_Symbol',	'Data_Value_Footnote', 'ClassID', 'QuestionID', 'TopicID', 'RowId', 'Data_Value_Alt', 'StratificationCategoryID1', 'StratificationCategoryID2', 'StratificationID1', 'StratificationID2', 'Data_Value_Alt', 'LocationAbbr', 'LocationID', 'Geolocation'], axis=1)

df['Stratification2'] = df['Stratification2'].str.replace(',', '', regex=False)
df['DataValueTypeID'] = df['DataValueTypeID'].str.replace(',', '', regex=False)
df['Question'] = df['Question'].str.replace(',', '', regex=False)
df['Question'] = df['Question'].str.replace('"', '', regex=False)

# Replace missing values 
for column in df.columns:
    if df[column].isnull().sum() == 0:
        continue
    if df[column].dtype in ['float64', 'int64']:
        df[column] = df.groupby(clas)[column].transform(lambda x: x.fillna(x.mean()))
    else:
        df[column] = df.groupby(clas)[column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))

# Normalize Valeues 
norm = ['Data_Value', 'Low_Confidence_Limit', 'High_Confidence_Limit']
for val in norm: 
    df[val] = zscore(df[val])

for column in df.columns:
    if column == 'Class':
        continue
    if df[column].dtype not in ['float64', 'int64']:
        df[column] =  pd.factorize(df[column])[0] + 1

# Saving Data 
df = df.reset_index(drop=True)

df.to_csv("Alzh_no_na.csv")

df = pd.read_csv("Alzh_no_na.csv", index_col=0)
print(df.head())

# Splitting Data
X_train, X_temp, y_train, y_temp = train_test_split( df.drop(clas, axis=1), df[clas], stratify=df[clas], test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.33, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(len(train_df), len(val_df), len(test_df))
train_df.to_csv("Alzh_train.csv")
val_df.to_csv("Alzh_val.csv")
test_df.to_csv("Alzh_test.csv")
