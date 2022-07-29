#!/usr/bin/env python
from pandas import *
import shutil
import os

data = read_csv("/home/user/Desktop/recon/code/index_dat2.csv")
name=data['name2'].tolist()

source = "/home/user/Desktop/recon/code/data/domain_os/"
destination = "/home/user/Desktop/recon/code/validate_recon/"
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
