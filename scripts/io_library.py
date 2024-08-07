from datetime import datetime
import pandas as pd
import os
import sys
import socket
import pickle
import h5py
import numpy as np
import random

input_dir = None
output_dir = None


class MyLib:
    # input_dir = ''
    # output_dir = None

    @staticmethod
    def get_time():
        return datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    @staticmethod
    def load_csv(file_path, **kwargs):
        """
            sample argument:  **{'header': None})
        """
        try:
            data_df = pd.read_csv(os.path.join(input_dir, file_path), **kwargs)
            print(os.path.join(input_dir, file_path), " is loaded, shape: ", data_df.shape)
            print()
            return data_df
        except OSError:
            print("Could not open/read file:", os.path.join(input_dir, file_path))
            sys.exit()

    @staticmethod
    def load_xls(file_path, sheet_no, **kwargs):
        """
            sample argument:  **{'header': None})
        """
        try:
            supp = pd.ExcelFile(os.path.join(input_dir, file_path))
            data_df = supp.parse(supp.sheet_names[sheet_no], **kwargs)
            # print(os.path.join(root_path, file_path), " is loaded, shape: ", data_df.shape)
            print()
            return data_df
        except OSError:
            print("Could not open/read file:", os.path.join(input_dir, file_path))
            sys.exit()

    @staticmethod
    def save_pkl(data, file_path):
        with open(os.path.join(output_dir, file_path), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('\nFile ', os.path.join(output_dir, file_path), 'saved, data size: ', len(data))
            print()

    @staticmethod
    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print('\nFile ',  file_path, " is loaded, data type:", type(data),
                  " size: ", len(data))
            print()
            return data

    @staticmethod
    def load_tsv(file_path, **kwargs):
        try:
            data_df = pd.read_csv(os.path.join(input_dir, file_path), delimiter="\t", low_memory=False, **kwargs)
            # data_df = pd.read_csv('input_files/' + file_path, delimiter="\t")
            print('input_files/' + file_path, " is loaded, shape: ", data_df.shape)
            print()
            return data_df
        except OSError:
            print("Could not open/read file:", 'input_files/' + file_path)
            sys.exit()


    @staticmethod
    def save_csv(data_df, file_path, **kwargs):
        try:
            data_df.to_csv(os.path.join(output_dir, file_path), **kwargs)
            print()
            print('File ', os.path.join(output_dir, file_path), 'saved, shape:', data_df.shape)
        except OSError:
            print("Could not find folder:", os.path.join(output_dir, file_path))
            sys.exit()

    @staticmethod
    def load_h5py(file_path):
        src = h5py.File(file_path, 'r')
        try:
            dim_0 = [x.decode('utf8') for x in src['dim_0']]
            dim_1 = [x.decode('utf8') for x in src['dim_1']]
            data = np.array(src['data'])
            print(file_path, " is loaded, shape: ", data.shape)
            print()
            return pd.DataFrame(index=dim_0, columns=dim_1, data=data)
        finally:
            src.close()

    @staticmethod
    def save_hdf5(data_df, file_path):
        dest = h5py.File(os.path.join(output_dir, file_path), 'w')

        try:
            dim_0 = [x.encode('utf8') for x in data_df.index]
            dim_1 = [x.encode('utf8') for x in data_df.columns]

            dest.create_dataset('dim_0', track_times=False, data=dim_0)
            dest.create_dataset('dim_1', track_times=False, data=dim_1)
            dest.create_dataset("data", track_times=False, data=data_df.values)
            print('\nFile ', os.path.join(output_dir, file_path), 'saved, data shape: ', data_df.shape)
            print()
        finally:
            dest.close()