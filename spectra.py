from unittest import result
from matplotlib.pyplot import cla
from nbformat import read
import numpy as np
import pandas as pd
import os
import glob
import json
import requests
import io
import string
import subprocess
import shutil
import tempfile
import gzip
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tqdm import tqdm


not_acceptable_chars = 'abcdfghijklmnopqrstuvwxyzABCDFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*/:,;<=>?@[\\]^_`{|}~'

PEPSI_SAXS_PATH = "./Pepsi_SASX"

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class RcordLine(object):
    """
    Class used to store a single SAXS measurement from any type of database
    """
    def __init__(self, q_vector, intentsities, I0 = None , R_g = None, D_max = None, MM = None) -> None:
        assert len(q_vector) == len(intentsities), 'q and I should have same number of points'
        assert len(q_vector) > 0, 'cannot instatiate with no data'
        self.q_vector = q_vector
        self.intensities = intentsities
        self.R_g = R_g
        self.D_max = D_max
        self.MM = MM
        self._q_min = q_vector[0]
        self._q_max = q_vector[-1]
        if self._q_min == 0:
            self.I0 = intentsities[0]
        else:
            self.I0 = I0
        #if self.R_g is None:
        #    self.guinier_analysis()
    def guinier_analysis(self):
        assert len(self.intensities) >=20, 'In order to a have a good Guinier plot fit, we need at least 20 points'
        model = LinearRegression()
        q2 = self.q_vector**2
        logI = np.log(self.intensities)
        model.fit(q2[0:20].reshape(-1, 1), logI[0:20].reshape(-1, 1))
        self.I0 = np.exp(model.intercept_)
        self.R_g = np.sqrt(-3*model.coef_)

    def get_normalized_krathky(self):
        assert self.I0 is not None, 'IO needs to be determined'
        assert self.R_g is not None, 'Rg has not been determined yet'
        krathky_intensisties = (self.q_vector*self.R_g)**2*self.intensities/float(self.I0)
        krathky_q = self.q_vector*self.R_g
        return krathky_q, krathky_intensisties

    def save_RcordLine(self, filepath):
        props = {'q':self.q_vector, 'Iq':self.intensities, 'Rg':self.R_g, 'Dmax':self.D_max, 'MM':self.MM}
        with open(filepath, 'r') as fp:
            json.dump(props, fp)
    
    def load_from_record(self, filepath):
        with open(filepath, 'r') as fp:
            record_properties = json.load(fp)
            self.q_vector = record_properties['q']
            self.intensities = record_properties['Iq']
            self.R_g = record_properties['Rg']
            self.D_max = record_properties['Dmax']
            self.MM = record_properties['MM']      

    def extract_tf_features(self):
        if self.MM or self.D_max is None or self.MM is None:
            self.R_g = np.nan
            self.D_max = np.nan
            self.MM = np.nan
        features = {
                "Rg": tf.train.Feature(float_list=tf.train.FloatList(value=[self.R_g])),
                "D_max": tf.train.Feature(float_list=tf.train.FloatList(value=[self.D_max])),
                "MM": tf.train.Feature(float_list=tf.train.FloatList(value=[self.MM])),
                "q": tf.train.Feature(float_list=tf.train.FloatList(value=self.q_vector)),
                "I": tf.train.Feature(float_list=tf.train.FloatList(value=self.intensities))
            }
        return features



class SASDBRearder(object):
    """
    class that fetches a SAXS curve from the SAXSBD dataset
    """
    def __init__(self, molecule_code) -> None:
        self._molecule_code = molecule_code
        self._code_summary = requests.get("https://www.sasbdb.org/rest-api/entry/summary/{}/".format(self._molecule_code))
    
    def url_to_vectors(self, url):
        q_vec = []
        I_vec = []
        f = requests.get(url)
        text_data = f.text
        for line in text_data.split("\n"):
            line = line.strip()
            if len(line) >0:
                #print(line)
                #print(not_acceptable_chars)
                has_any = any([char in line for char in not_acceptable_chars])
                if not has_any:
                    data_line= [float(data) for data in line.split() if data not in {'', '--'}]
                    if len(data_line) > 1:
                        q_vec.append(data_line[0])
                        I_vec.append(data_line[1])
        assert len(I_vec) > 0
        return np.array(q_vec), np.array(I_vec)
    
    def get_record(self) -> RcordLine:
        I_url = self._code_summary.json()['intensities_data']
        I0 = self._code_summary.json()['guinier_i0']
        Rg = self._code_summary.json()['guinier_rg']
        D_max = self._code_summary.json()['pddf_dmax']
        MM = self._code_summary.json()['porod_mw']
        q, I = self.url_to_vectors(I_url)

        return RcordLine(q, I, I0, Rg, D_max, MM)


class PDBReader(object):
    """
    class that fetches a .pdb file stored locally and builds a record by simulating 
    a SAXS diffusion curve using the Pepsi-SAXS package
    """
    def __init__(self, pdb_directory, protein_code, sim_directory) -> None:
        self._path_to_pdb = os.path.join(pdb_directory, protein_code+'.pdb.gz')
        self._sim_directory = sim_directory
        self.protein_code = protein_code
    
    def generate_simulated_curve(self):
        with gzip.open(self._path_to_pdb, 'rb') as f_in:
            with tempfile.NamedTemporaryFile() as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.flush()
                pepsi_log = subprocess.run(["./Pepsi-SAXS", f_out.name, "-o", './{}/{}.dat'.format(self._sim_directory, self.protein_code)], 
                                            stdout=subprocess.PIPE)
    
    def get_record(self):
        sim_path = self._sim_directory + '/{}.dat'.format(self.protein_code)
        log_path = self._sim_directory + '/{}.log'.format(self.protein_code)
        df = pd.read_csv(sim_path, skiprows=5, delim_whitespace=True, usecols=[0,1])
        df.columns = ["q", "Iq"]
        q = df['q'].to_numpy()
        Iq = df['Iq'].to_numpy()
        I0 = Iq[0]

        with open(log_path, 'rb') as log_file:
            log_text = log_file.read().decode()
        
        for line in log_text.split("\n"):
            if line.startswith("Radius of gyration."):
                Rg = float(line.split(':')[1].strip().split(" ")[0])
            if line.startswith("Total molecular weight"):
                MM = float(line.split(':')[1].strip().split(" ")[0])
            if line.startswith("Maximum extension from the center"):
                D_max = float(line.split(':')[1].strip().split(" ")[0])
            
        return RcordLine(q, Iq, I0, Rg, D_max, MM)


def unzipped_name(file_name):
    unzipped  = file_name.split('/')[-1].split('.')[0]
    return unzipped

def main():
    pdp_directory = './pdb_data/'
    sim_directory = './pdb_saxs'

    x = requests.get("https://www.sasbdb.org/rest-api/entry/codes/all/")
    codes = x.json()
    sasdb_record_readers = [SASDBRearder(code['code']) for code in codes if code['code'] not in ('SASDB78', 'SASDFP5')]

    with tf.io.TFRecordWriter('./sasdb.tfrecord') as tfrecord:
        for reader in tqdm(sasdb_record_readers):
            record = reader.get_record()
            features = record.extract_tf_features()
            example = tf.train.Example(features=tf.train.Features(feature=features))
            tfrecord.write(example.SerializeToString())
    
    list_of_files = glob.glob('./pdb_data/*.pdb.gz')
    prot_ids = [unzipped_name(filename) for filename in list_of_files]
    pdb_readers = [PDBReader(pdb_directory=pdp_directory, protein_code=prot_id, sim_directory=sim_directory) for prot_id in prot_ids]
    
    with tf.io.TFRecordWriter('./pdb.tfrecord') as tfrecord:
        for reader in pdb_readers:
            try:
                reader.generate_simulated_curve()
                record = reader.get_record()
                features = record.extract_tf_features()
                example = tf.train.Example(features=tf.train.Features(feature=features))
                tfrecord.write(example.SerializeToString())
            except EOFError:
                continue            

    
if __name__ == '__main__':
    main()


