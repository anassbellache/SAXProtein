#!/usr/bin/env python

import sys
import os
import numpy as np
import shutil
import datetime
from datetime import date


image_dir = '../test'
Paths=[]
fichiers = []
for root, dirs, files in os.walk(image_dir):
    fichiers.append(root)
#fichiers
Paths=[]
for image_dir in fichiers:
    for root, dirs, files in os.walk(image_dir):
        if len(files):
            for file in files:
                if file.endswith("dat"):
                    path=os.path.join(root,file)
                    Paths.append(path)
files = []
for file in np.unique(Paths):
    if "QDetector" in file:
        files.append(file)

filename = files[0]

texte=[]
with open(filename, "r") as f_read:
    for elt in f_read.readlines():
        text = ' '.join(elt.split())
        texte.append(text)

for i in range(len(texte)):
    if "variables: q I I_err N" in texte[i]:
        tronc = texte[i+1:]

Data=[]
for j in [0,1,2,3]:
    DD=[]
    for i in tronc:
        try:
            DD.append(float(i.split()[j]))
            a=float(i.split()[j])
        except:
            DD.append(a)
    Data.append(DD)

kratky=["kratky"]
for i in range(len(Data[0])):
    if Data[1][i]==0:
        kratky.append(np.nan)
    else:
        kratky.append(Data[0][i]**2*Data[2][i]/Data[1][i])
    
    
Q=["Q"]
for i in range(len(Data[0])):
    Q.append(Data[0][i])

I=["I"]
for i in range(len(Data[0])):
    I.append(Data[1][i])

Ie=["I_err"]
for i in range(len(Data[0])):
    Ie.append(Data[2][i])
    
N=["N"]
for i in range(len(Data[0])):
    N.append(Data[3][i])
    
model=sys.argv[1]
today = date.today()
datee=str(today)[:4]+"_"+str(today)[5:7]+"_"+str(today)[8:]
time=str(datetime.datetime.now())[-5:]
Time=datee+"_"+time

os.chdir("../")
image_dir ='./DATA_test'
np.savetxt(image_dir+"/"+model+"_"+Time+".csv", [p for p in zip(Q,I,Ie,kratky,N)],
          delimiter=',', fmt='%s')
os.chdir("test")
n=filename.index(']')
file = filename[:n+1]
shutil.rmtree(file)