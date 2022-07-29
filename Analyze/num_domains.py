#This is used to find the number of domains and oses in our data 
#!/usr/bin/env python
import os 
import numpy as np
source="" #path of domain_os folder
domains=[]
oses=[]
for file in os.listdir(source):
   if(file=="general.json"):continue
   f=file.split("_")
   print(f)
   domains.append(f[0])
   oses.append(f[1])
domains=np.unique(domains)
oses=np.unique(oses)
print(domains)
print(len(domains))
print(oses)
print(len(oses))
