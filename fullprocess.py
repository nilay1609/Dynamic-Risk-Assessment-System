import training
import scoring
import deployment
import diagnostics
import reporting
import os, json, glob
import ingestion
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
input_folder_path = os.path.join(config['input_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
ingested = open(prod_deployment_path + '/' + "ingestedfiles.txt", "r")
latest_score = open(prod_deployment_path + '/' + 'latestscore.txt', "r")
latest_score = latest_score.read()
latest_score = int(float(latest_score))
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

all_files = glob.glob(input_folder_path + "/*.csv")

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

if all_files not in ingested:
    os.system('python ingestion.py')
    os.system('python training.py')
    os.system('python scoring.py')
    score = open(model_path + '/' 'latestscore.txt', "r")
    score = score.read()
    score = int(float(score))
    if score > latest_score:
        os.system('python deployment.py')
    else:
        print('No model drift')
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model





