#!/usr/bin/env python
import sys
import os
import numpy as np
import datetime


class McTraceConfigurator:
    def __init__(self, image_dir = './test') -> None:
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
                        if file.endswith("instr"):
                            path=os.path.join(root,file)
                            Paths.append(path)
        files = np.unique(Paths)
        self.instr_file = files[0]

    def prepare_intrument(self, model, param):
        with open(self.instr_file, "r") as f_read:
            t = f_read.read()
            #model = sys.argv[1]#input("Model Index : ")
            f = t.replace("%MODEL_INDEX%", model)

        with open(self.instr_file, "w") as f_write:
            f_write.truncate()
            f_write.write(f)
            f_write.close()

        with open(self.instr_file, "r") as f_read:
            t = f_read.read()
            #param = sys.argv[2]#input("Parametre du modéle : ")
            f = t.replace("%MODEL_PARS%", param)

        with open(self.instr_file, "w") as f_write:
            f_write.truncate()
            f_write.write(f)
            f_write.close()