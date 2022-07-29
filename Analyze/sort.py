#This is used to create the training and testing folders for predictions 
#!/usr/bin/env python
from pandas import *
import shutil
import os

data = read_csv("index_dat.csv") #csv ile with names of a the data files
name=data['name'].tolist()

source = "/domain_os/" #source directory
destination = "/validate/" #destination directory
i=0
for filename in os.listdir(source):
  
  
    if filename not in name:
      i=i+1
      
      t=source +filename
      d=os.path.join(destination,filename)
      fp = open(d, 'w')
      d=destination +filename
      shutil.copyfile(t,d)
print(i)   
