import os
import json
def j(file):  
 # Opening JSON file
 f = open(file)
 # returns JSON object as 
 # a dictionary
 data = json.load(f)
 return data
def classes(f_path):
 final_list=[]
 source=f_path
 for filename in os.listdir(source):
  f=source+filename
  
  data=j(f)
  for fruit in data.keys():
   for fruit1 in data.get(fruit).keys():
    if(fruit1 == "pii_types" and data.get(fruit).get(fruit1) != "null"):
      if(data.get(fruit).get(fruit1)!=None):
       list=data.get(fruit).get(fruit1)
        final_list.extend(data.get(fruit).get(fruit1))
