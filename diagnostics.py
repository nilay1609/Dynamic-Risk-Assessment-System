
import pandas as pd
import numpy as np
import timeit
import os
import json
import glob
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    df = pd.read_csv(data)
    filename = 'trainedmodel' + '.pkl'
    with open(model_path + '/' + filename, 'rb') as file:
        model = pickle.load(file)
    x = df.iloc[:,1:4]
    y = df.iloc[:,-1]
    predicted = model.predict(x)
    predicted.tolist()
    return predicted #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    all_files = glob.glob(dataset_csv_path + "/*.csv")
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.drop_duplicates(inplace = True)    
    l = []
    l.append(np.mean(df['lastmonth_activity']))
    l.append(np.mean(df['lastyear_activity']))
    l.append(np.mean(df['number_of_employees']))
    l.append(np.mean(df['exited']))
    l.append(np.median(df['lastmonth_activity']))
    l.append(np.median(df['lastyear_activity']))
    l.append(np.median(df['number_of_employees']))
    l.append(np.median(df['exited']))
    l.append(np.std(df['lastmonth_activity']))
    l.append(np.std(df['lastyear_activity']))
    l.append(np.std(df['number_of_employees']))
    l.append(np.std(df['exited']))
    
    return l #return value should be a list containing all summary statistics
  
def missing_data():
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    na_prcnt = df.isnull().mean().tolist()
    return na_prcnt

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime_ing = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing_ing=timeit.default_timer() - starttime_ing
    starttime_tr = timeit.default_timer()
    os.system('python3 training.py')
    timing_train=timeit.default_timer() - starttime_tr
    li = list()
    li.append(timing_ing)
    li.append(timing_train)
    return li #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    package_list = subprocess.check_output(['pip', 'list', '--outdated'])
    return package_list


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
