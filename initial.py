import pandas as pd 
from tqdm import tqdm 
from scipy import stats
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

df = pd.read_csv('Alzheimer_s_Disease_and_Healthy_Aging_Data.csv')

print(df['Class'].value_counts())

#use notna() method
result = df.notna().values.all()

#print the result
print(result)
print(df.isnull())
total = df.isnull().stack()[df.isnull().stack()]
print(len(total))

#Read the entire file content
file_path = './Alzh_remove.csv'

with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

file_content = file_content.replace('"', '')

with open('./Data.csv', 'w', encoding='utf-8') as file:
    file.write(file_content)
