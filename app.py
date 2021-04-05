from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from flask import jsonify, make_response
import pickle
#mport create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import json
import os
from diagnostics import *
from scoring import *

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filepath = request.args.get('filepath')
    predicted = model_predictions(filepath).tolist()
    response = app.response_class(
        response=json.dumps(predicted),
        status=200,
        mimetype='application/json'
    )
    return response #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1 = score_model()
    response = app.response_class(
        response=json.dumps(f1),
        status=200,
        mimetype='application/json'
    )
    return response #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stat = dataframe_summary()
    
    return jsonify(stat) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    time = execution_time()
    return jsonify(time) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
