#To create the json file for loading the specific domains of training data 
import os
import json
from pandas import *
source="source/" #path to source directory 
f="/content/index_dat.json" #json file to be changed (  This file contains data of all the training files)
f1=open("/content/index_dat.json")
#list=os.listdir(source)
d = read_csv("/content/index_dat.csv") #csv file with details of the domain_os to be loaded 
list=d['name'].tolist()

data=json.load(f1)
data2=data.copy()
for key in data.keys():
 
 if key not in list:
   del data2[key]
os.remove(f)
print(len(data2))
with open(f, 'w') as f:
    json.dump(data2, f, indent=4)  
