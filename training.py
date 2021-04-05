from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn import preprocessing

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    X = df.iloc[:,1:4]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    #use this logistic regression for training
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    #fit the logistic regression to your data
    model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train,y_train)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    filename = 'trainedmodel' + '.pkl'
    pickle.dump(model, open(model_path + '/' + filename, 'wb'))
    
    


if __name__ == '__main__':
    train_model()