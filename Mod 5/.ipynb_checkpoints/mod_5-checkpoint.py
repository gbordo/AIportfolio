import pandas as pd 
import numpy as np 
import os

data = pd.read_csv(os.getcwd()+'/cars.csv')
data.columns

# The model name has many model names but cannot be bined becuase you would need to account for many of the model names in order to account for the majority of the data and the deep learning model should be able to run ~1200 columns once we hot-encode so I will not bin model_name

(data['model_name'].value_counts()/data['model_name'].value_counts().sum()).iloc[0:100].sum()

# All of the object and boolean variables are nominal categorical. Order does not matter so I will assign One-hot Encoding. For the boolean variables the True will be replaced with 1 and the False will be replaced with 0. 

#Gather Variables that are object or bool 
cat_var_obj=[]
cat_var_bool=[]
for col in data.columns: 
    if data[col].dtype in [object,bool]:
        if data[col].dtype==object:
            cat_var_obj.append(col)
        elif data[col].dtype==bool:
            cat_var_bool.append(col)
            
#Get the one-hot encoding for all object colums
dumm = pd.get_dummies(data[cat_var_obj])

#Encode the boolean variables as (0,1)
data_bool = data[cat_var_bool].apply(lambda x: x.astype('category'))
data_bool = data_bool[cat_var_bool].apply(lambda x: x.cat.codes)

cat_var = pd.merge(dumm,data_bool,left_index=True, right_index=True)
transformed_cols = cat_var_obj + cat_var_bool
other_variables = [ i for i in data.columns if i not in transformed_cols]

train_data = pd.merge(cat_var,data[other_variables],left_index=True, right_index=True)

print("Sample of all training data after converting categorical variables: \n")
print(train_data)