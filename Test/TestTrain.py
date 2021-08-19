import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('dumd.csv')

x_train, x_test, y_train, y_test = train_test_split( data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)

x_train['y'] = y_train
x_test['y']= y_test

#x_train.to_csv (r'x_train.csv', index = False, header=True)
x_test.to_csv (r'dumd_test.csv', index = False, header=True)

shuffled=x_train.sample(frac=1)
result=np.array_split(shuffled,3)

for i in range(len(result)):
    result[i].to_csv (r'dumd_train'+str(i+1)+'.csv', index = False, header=True)
