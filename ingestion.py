import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
f= open(output_folder_path + '/' + 'ingestedfiles.txt',"w+")


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    all_files = glob.glob(input_folder_path + "/*.csv")
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.drop_duplicates(inplace = True)
    filename = 'finaldata' + '.csv'
    concatenated_df.to_csv(output_folder_path + '/' + filename,index=False)
    for i in all_files:
        f.write("%s " % i)
    f.close()


if __name__ == '__main__':
    merge_multiple_dataframe()
