#!/usr/bin/env python

from statistics import mode
import sys
import os
import numpy as np
import shutil
import datetime
from datetime import date



class McTraceRecord:
    def __init__(self, model, image_dir = './[model]') -> None:
        self.model = model
        Paths = []
        fichiers = []
        for root, dirs, files in os.walk(image_dir):
            fichiers.append(root)
        #fichiers
        Paths = []
        for image_dir in fichiers:
            for root, dirs, files in os.walk(image_dir):
                if len(files):
                    for file in files:
                        if file.endswith("dat"):
                            path = os.path.join(root,file)
                            Paths.append(path)
        files = []
        for file in np.unique(Paths):
            if "QDetector" in file:
                files.append(file)
        #print("files: ", files[0])
        self.q_detector_file = files[0]

    def parse_detector_curve(self):
        texte = []
        with open(self.q_detector_file, "r") as f_read:
            for elt in f_read.readlines():
                text = ' '.join(elt.split())
                texte.append(text)

        for i in range(len(texte)):
            if "variables: q I I_err N" in texte[i]:
                tronc = texte[i+1:]

        Data = []
        for j in [0,1,2,3]:
            DD = []
            for i in tronc:
                try:
                    DD.append(float(i.split()[j]))
                    a = float(i.split()[j])
                except:
                    DD.append(a)
                Data.append(DD)

        self._kratky = ["kratky"]
        for i in range(len(Data[0])):
            if Data[1][i]==0:
                self._kratky.append(np.nan)
            else:
                self._kratky.append(Data[0][i]**2*Data[2][i]/Data[1][i])
    
    
        self._Q = ["Q"]
        for i in range(len(Data[0])):
            self._Q.append(Data[0][i])

        self._I = ["I"]
        for i in range(len(Data[0])):
            self._I.append(Data[1][i])

        self._Ie = ["I_err"]
        for i in range(len(Data[0])):
            self._Ie.append(Data[2][i])
    
        self._N = ["N"]
        for i in range(len(Data[0])):
            self._N.append(Data[3][i])
    
#model = sys.argv[1]

    def save_model(self, image_dir = '../DATA'):
        today = date.today()
        datee = str(today)[:4]+"_"+str(today)[5:7]+"_"+str(today)[8:]
        time = str(datetime.datetime.now())[-5:]
        Time = datee+"_"+time
        #os.chdir("../")
        np.savetxt(image_dir+"/"+self.model+"_"+Time+".csv", [p for p in zip(self._Q, self._I, self._Ie, self._kratky, self._N)],
        delimiter = ',', fmt = '%s')
        #os.chdir("test")

        print(os.path.dirname(self.q_detector_file))
        #n = self.q_detector_file.index(']')
        sim_location = './[model]'
        shutil.rmtree(sim_location)