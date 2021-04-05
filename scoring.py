from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import glob


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 



#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    all_files = glob.glob(test_data_path + "/*.csv")
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    test_df   = pd.concat(df_from_each_file, ignore_index=True)
    filename = 'trainedmodel' + '.pkl'
    with open(model_path + '/' + filename, 'rb') as file:
        model = pickle.load(file)
    x = test_df.iloc[:,1:4]
    y = test_df.iloc[:,-1]
    predicted = model.predict(x)
    f1score = f1_score(predicted,y)
    f= open(model_path + '/' + 'latestscore.txt',"w+")
    f.write(str(f1score))
    f.close()
    return f1score
if __name__ == '__main__':
    score_model()