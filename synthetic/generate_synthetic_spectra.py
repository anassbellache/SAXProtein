#!/usr/bin/env python


import pandas as pd
import os 
import datetime
from datetime import date
import shutil
import os 
import datetime
from datetime import date
import shutil
import codecs 
import random as rd
from Script_index import McTraceConfigurator
from make_csv import McTraceRecord

init='python init.py'
os.system(init)




file = "test/sasview.html"  
files = codecs.open(file, 'r', "utf-8") 
a=files.read()
model_index=[]
m=[]
model=[]
v=[]
v1=[]
v2=[]
b=a.split("Units")[1]
e=b.split("Name convention")[0]
c=e.split("<tr>")
for i in range(5,len(c)-1):
    f=c[i]#[7]
    par=f.split('<td>')
    v.append(par[2].split('<')[0])
    v1.append(par[3].split('<')[0])
    v2.append(par[4].split('<')[0])
    
c=0
com=[]
for i in v:
    if i[0]==' ' or i=='Parameters:' or i[0]=='&' or i[0]=='{':
        com.append(c)
    c=c+1

for i in com:
    v[i]='#'
    
d1 = {'model_': v, 'parameter_': v1, 'inter_': v2}
df1 = pd.DataFrame(data=d1)
df1.drop(df1[df1['model_']=='#'].index, inplace=True)

df1 = df1.reset_index()


file = "test/sasview.html"  
files = codecs.open(file, 'r', "utf-8") 
a=files.read()
model_index=[]
m=[]
model=[]

for i in range(1,59):
    model_index.append(a.split("(")[i].split(")")[0])
model_degre=[]
model_simple=[]
for i in range(58):
    if ("theta" or "phi" or "psi") in model_index[i]:
        model_degre.append(model_index[i])
    else:
        model_simple.append(model_index[i])
############################################################
a = open("test/note.txt","r")
a = a.read()
m = []
model = []
for i in a.split("\n"):
    m.append(i.split("\t")[0])
for i in m:
    if len(i) >= 3:
        model.append(i)

aa = a.split("{")
t=[]
for i in aa:
    if '}' in i:
        t.append(i)
tt=[]
for i in t:
    tt.append(i.split('}')[0])
    
D=[0]
for i in tt:
    x=[]
    for j in i.split(","):
        #print(j)
        x.append(float(j.split()[0]))
    D.append(x)


d = {'model': model[1:], 'parameter': D[1:], 'model_index': model_index}
df = pd.DataFrame(data=d)


import numpy as np

at=[]
for i in range(df.shape[0]):
    n=len(df['parameter'][i])
    at.append(n)
    

lig=at[::2]
col=np.cumsum(at[1::2])

az1_par=[df1['parameter_'][:5]]
az2_par=[df1['parameter_'][:7]]
az1_int=[df1['inter_'][:5]]
az2_int=[df1['inter_'][:7]]

for i in range(1,len(col)):
    az1_par.append(df1['parameter_'][col[i-1]:col[i-1]+lig[i]])
    az1_int.append(df1['inter_'][col[i-1]:col[i-1]+lig[i]])
    #print(f"{col[i-1]}-----{lig[i]}")
for i in range(len(col)-1):   
    az2_par.append(df1['parameter_'][col[i]:col[i+1]])
    az2_int.append(df1['inter_'][col[i]:col[i+1]])
    #print(f"{col[i]}-----{col[i+1]}")
    
az_par=[]
for i,j in zip(az1_par,az2_par):
    az_par.append(i)
    az_par.append(j)
    
az_int=[]
for i,j in zip(az1_int,az2_int):
    az_int.append(i)
    az_int.append(j)
    
var=[]
inter=[]
for j in range(len(az_par)):
    a=[]
    for i in az_par[j]:
        a.append(i)
    var.append(a)
for j in range(len(az_int)):
    a=[]
    for i in az_int[j]:
        a.append(i)
    inter.append(a)
    
df['Variable'] = var
df['Intervalle'] = inter



def sim_para(par_indx,par):
    
    if par_indx=='[-360, 360]':
        res=rd.randint(-180,180)
    
    elif par_indx=="[-inf, inf]":
        if abs(par)<1:
            par=1
            
        res=rd.randint(-int(par/2)+round(par),int(par/2)+round(par))
        
    elif par_indx=='[0, 0.74]':
        res=0.74*rd.random()
        
    elif par_indx=='[0, inf]':
        if abs(par)<1:
            par=1
            
        res=rd.randint(0,int(par/2)+round(par))
        
    elif par_indx=='[0.0, inf]':
        if abs(par)<1:
            par=1
            
        res=rd.randint(0,int(par/2)+round(par))
        
    elif par_indx=='[0.0, 0.8]':
        res=0.8*rd.random()
        
    elif par_indx=='[0.01, 0.1]':
        res=abs(0.1*rd.random()-0.01*rd.random())
        
    elif par_indx=='[1.0, 6.0]':
        res=rd.randint(1,6)
        
    elif par_indx=='[1e-16, 6.0]':
        res=rd.randint(0,6)
        
    return(res)


def parame(liste,index):
    #print(liste)
    par=liste[index]
    par_indx=df['Intervalle'][index-1]
    parametre={}
    for j in range(1,1000):
        result=[]
        for i in range(len(par)):
            result.append(sim_para(par_indx[i],par[i]))
        #print(result)
        DD=result
        #for j in range(1,50):
        a=str(DD)
        a=a.replace(a[0],'{')
        a=a.replace(a[-1],'}')
        c=a.split(", ")
        b=','.join(c)
        parametre[j]=b
    return parametre



index=int(input("Model Index : "))
parametre = parame(D,index)

for i in range(1,10):
    shutil.copy('C:/Users/mouha/Desktop/TEST_IA/script_orig/templateSasView1.instr', 'test')
    os.chdir("test")
    m=parametre[i]
    print(m)
    #cmd = "python Script_index.py "+str(index)+' '+m
    cmd1 = 'mxrun & templateSasView1 lambda=1.54 dlambda=0.05 -d[model]'
    #cmd2 = "python make_csv.py "+model[index]


    #os.system(cmd)
    os.system(cmd1)
    #os.system(cmd2)
    os.chdir("../")
    
#os.chdir("test")
#file = os.getcwd()
#shutil.rmtree(file)