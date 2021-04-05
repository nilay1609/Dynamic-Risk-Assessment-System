from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil
import errno


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_deploy():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    try: 
        shutil.copytree(dataset_csv_path + '/' + 'ingestedfiles.txt', prod_deployment_path) 
    except OSError as err: 
  
        # error caused if the source was not a directory 
        if err.errno == errno.ENOTDIR: 
            shutil.copy2(dataset_csv_path + '/' + 'ingestedfiles.txt', prod_deployment_path) 
        else: 
            print("Error: % s" % err)
    try: 
        shutil.copytree(model_path + '/' + 'latestscore.txt', prod_deployment_path) 
    except OSError as err: 
  
    # error caused if the source was not a directory 
        if err.errno == errno.ENOTDIR: 
            shutil.copy2(model_path + '/' + 'latestscore.txt', prod_deployment_path) 
        else: 
            print("Error: % s" % err)    
    try: 
        shutil.copytree(model_path + '/' + 'trainedmodel.pkl', prod_deployment_path) 
    except OSError as err: 
  
        # error caused if the source was not a directory 
        if err.errno == errno.ENOTDIR: 
            shutil.copy2(model_path + '/' + 'trainedmodel.pkl', prod_deployment_path) 
        else: 
            print("Error: % s" % err)    

if __name__ == '__main__':
    store_model_into_deploy()