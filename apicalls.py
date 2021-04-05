import requests
import json
import os
import subprocess
from subprocess import DEVNULL, STDOUT, check_call
from subprocess import Popen, PIPE

with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

#Call each API endpoint and store the responses
response1 = subprocess.run(['curl', 'http://127.0.0.1:5000/prediction?filepath=testdata/testdata.csv'],capture_output=True).stdout
response2 = subprocess.run(['curl', 'http://127.0.0.1:5000/scoring'],capture_output=True).stdout
response3 = subprocess.run(['curl', 'http://127.0.0.1:5000/summarystats'],capture_output=True).stdout
response4 = subprocess.run(['curl', 'http://127.0.0.1:5000/diagnostics'],capture_output=True).stdout



#combine all API responses
responses = {
    'response1': response1,
    'response2': response2,
    'response3': response3,
    'response4': response4
}


#write the responses to your workspace
with open(output_model_path + '/' + 'apireturns.txt',"w+") as f:
    print("{}".format(responses), file=f)
f.close()
