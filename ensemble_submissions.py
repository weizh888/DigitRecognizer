import os
import numpy as np
import pandas as pd

csv_path = './results'
allFiles = [file for file in os.listdir(csv_path) if file.endswith('.csv')]
#allFiles = ['aug_2_cycle.csv', 'submission_oldModel.csv', 'mnist-test5.csv', 'submission_0.4_relu.csv', 'submission_0.3_sigmoid.csv', 'mnist-test3.csv']
alltest = pd.DataFrame()

list_ = []
for f in allFiles:
    df = pd.read_csv(csv_path + '/' + f, index_col=None)
    list_.append(df['Label'])
alltest = pd.concat(list_, axis=1)
#print(alltest)
new_result = alltest.mode(axis=1)[0].astype(int) # Select only the first mode value, and convert to integer label.  
#print(new_result)
d = {'ImageId': range(1, len(new_result)+1), 'Label': new_result} # Create dataframe from dictionary.  
new_submission = pd.DataFrame(d)
new_submission.to_csv('./ensemble_result.csv', index=False, header=True)
