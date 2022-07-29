def accuracy(f):
 
 f=open(f,'r')
 lines=f.readlines()
 total=0
 label=0
 pii=0
 for line in lines:
  d='"domain"'+":"
  if d in line:
   total=total+1
   s1='"is_correctPII"'+":"+'"True"'
   s2='"is_correctLabel"'+":"+'"True"'
  
   if s1 in line :
    pii=pii+1
   if s2 in line:
    label=label+1
 print(total,label,pii)
 labelling_accuracy=(label/total)*100
 type_accuracy=(pii/total)*100
 print("Labelling Accuracy=",labelling_accuracy)
 print("PII type accuracy=",type_accuracy)
