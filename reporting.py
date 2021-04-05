import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    all_files = glob.glob(dataset_csv_path + "/*.csv")
    df = (pd.read_csv(f) for f in all_files)
    df   = pd.concat(df, ignore_index=True)
    predicted = model_predictions(dataset_csv_path + '/' + 'testdata.csv')
    X = df.iloc[:,1:4]
    y = df.iloc[:,-1]
    cf_matrix = confusion_matrix(y, predicted)
    fig = sns.heatmap(cf_matrix, annot=True)
    fig = fig.get_figure()
    fig.savefig(model_path + '/' + "confusionmatrix.png")

if __name__ == '__main__':
    score_model()
