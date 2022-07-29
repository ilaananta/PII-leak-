from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import numpy as np
import pandas as pd
import gc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
def load(f):

 with open(f, 'r') as fp:
    file_content = fp.readlines()


 def parse_row(line, len_row):
    line = line.replace('{', '').replace('}', '')

    row = np.zeros(len_row)
    d=line.split(',')
    if(len(d)!=1):
     for data in line.split(','):
         index, value = data.split()
         row[int(index)] = float(value)

    return row


 columns = []
 len_attr = len('@attribute')
 

# get the columns
 for line in file_content:
    if line.startswith('@attribute '):
        col_name = line[len_attr:].split()[0]
        columns.append(col_name)
 rows = []
 len_row = len(columns)
# get the rows
 for line in file_content:
    if line.startswith('{'):
        
        rows.append(parse_row(line, len_row))

 df = pd.DataFrame(data=rows, columns=columns)
 # give path of training (   ......Recon) or test file(/content/drive/MyDrive/arff_manual2/ ...our data)
 domain_os=f.replace("f_path","")  #path of arff files
 domain_os=domain_os.replace(".arff","")
 temp=domain_os.split("_")
 domain=temp[0]
 os=temp[1]
 fo=[]
 fo1=[]
 for i in range(0,len(df.index)):
   fo.append(domain)
   fo1.append(os)

 df.insert(1,"domain",fo)
 df.insert(2,"OS",fo1)

 return df


# making final data frame by combining all training dataframes (per domain)


# Combing columns

col=[]
rows=[]
for file in os.listdir("/arff/"):  #path of training arff files 
  #print(file)
  if(file=="general_all.arff"):continue
  s="/arff/"+file
  df1=load(s)
  col1=df1.columns
  col=np.concatenate((col,col1))
 
col=np.unique(col)
print(col)
print(len(col))



# Combining rows
for file in os.listdir("/arff/"):   #path of training arff files 
  if(file=="general_all.arff"):continue
  
  s="/arff/"+file   #path of training arff files 
  df1=load(s)
  col1=df1.columns
  for index, row in df1.iterrows():
   map={}
   for c in col1:
    map[c]=row[c]
   ro=[]
   for i in range(0,len(col)): 
    ro.append("0.0")
   for i in range(0,len(col)):
     t=col[i]
     if t in map:
       ro[i]=map[t]
   rows.append(ro)  
      
   
# Chnaging position for y column to last       
df = pd.DataFrame(data=rows, columns=col)
#df=df.fillna(0.0)
col2=col.copy
c=df.pop("PIILabel")
d=df.pop("domain")
e=df.pop("OS")
df.insert(len(df.columns), 'PIILabel', c)
df.insert(0, 'domain', d)
df.insert(1, 'OS', e)

#df.to_csv("datafinal.csv")  
#print(df["+ab_test_data"]) 

#function for feature selection 
def select_features(X_train, y_train, X_test):
 fs = SelectKBest(score_func=chi2, k=40)
 fs.fit(X_train, y_train)
 X_train_fs = fs.transform(X_train)
 X_test_fs = fs.transform(X_test)
 return X_train_fs, X_test_fs, fs
 
# Converting domain_os string from categorical to numerical values
def numberit(column,dataframe,pos):
  
  li=dataframe[column]
  li=np.unique(li)
  i=1
  ma={}
  for co in li:
   ma[co]=i
   i=i+1


  r=dataframe.pop(column)

  for i in range(0,len(r)):
   s=r[i]
   r[i]=ma[s]
 
  dataframe.insert(pos,column,r)
  return dataframe,ma


#making training set
# Creating dependent and independent features
df,ma1=numberit("domain",df,0)

df,ma2=numberit("OS",df,1)
#print(df)
X = df.iloc[:, :-1].values

#print(X)
y = df.iloc[:, -1].values


#making test data
# combining arff files for test dataset

# check if training features are present in test files then take its value else 0
rows=[]
number=0
for file in os.listdir("/arff_manual/"): #path to testing arff files 
 number=number+1
 print(number) 
 f="/arff_manual/"+file    #path to testing arff files 
 df1=load(f)
 col1=df1.columns
 for index, row in df1.iterrows():
   map={}
   for c in col1:
    map[c]=row[c]
   ro=[]
   for i in range(0,len(col)): 
    ro.append("0.0")
   for i in range(0,len(col)):
     t=col[i]
     if t in map:
       ro[i]=map[t]
   rows.append(ro)
 del df1
 gc.collect()    
    
      
   
      
df_test = pd.DataFrame(data=rows, columns=col)
c=df_test.pop("PIILabel")
d=df_test.pop("domain")
e=df_test.pop("OS")
df_test.insert(len(df_test.columns), 'PIILabel', c)
df_test.insert(0, 'domain', d)
df_test.insert(1, 'OS', e)
# check numerical values of domain_os column in training and assign same value in testing dataset
def numberit2(column,dataframe,map,pos):
  r=dataframe.pop(column)
  temp=len(map.keys())+1
  map2={}
  z=np.unique(r)
  for item in z:
   if item in map.keys(): continue
   else:
    map2[item]=temp
    temp=temp+1 
     
  for i in range(0,len(r)):
  
   p=r[i]
   if p in map:
    r[i]=map[p]
   else:   
    r[i]=map2[p]
  dataframe.insert(pos,column,r)  
  return dataframe  


#doing feature selection 

df_test=numberit2("domain",df_test,ma1,0)
df_test=numberit2("OS",df_test,ma2,1)
print(df_test) 

# seperating independent and dependent features in test set
X_test = df_test.iloc[:, :-1].values

y_test = df_test.iloc[:, -1].values

print(X_test)
print(y_test)

# select same features as in training set
X_train_fs, X_test_fs, fs = select_features(X, y, X_test)
print(X_train_fs)
print(X_test_fs)


X_train_fs = np.asarray(X_train_fs).astype(np.float32)
model2=Sequential()
model2.add(Dense(12,input_shape=(40,), activation='sigmoid'))
model2.add(Dense(40, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(X_train_fs, y, epochs=150, batch_size=10)
_, accuracy = model2.evaluate(X_train_fs, y)
print('Accuracy: %.2f' % (accuracy*100))


X_test_fs = np.asarray(X_test_fs).astype(np.float32)
Y=model2.predict(X_test_fs)
classes_x=np.round(Y)
_, test_acc = model2.evaluate(X_test_fs, y_test, verbose=0)
print(test_acc*100)
