# MIT License
#
# Copyright (c) 2023 Md Hasan Shahriar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Project: CANtropy - Time Series Feature Extraction-Based Intrusion Detection Systems for Controller Area Networks
# Author: Md Hasan Shahriar
# Email: hshahriar@vt.edu
#

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path


def generate_dataset(file_name, file_path, org_features):
  
    print("Generating dataset: ", file_name)
    extd_file_dir = Path(f"{file_path.parent}/generated/{file_name}_generated.csv")   
    # Load original dataset
    df_original = pd.read_csv(file_path, skiprows=1, names=org_features)
    # Replace 'id' from the ID number only in SynCAN dataset
    df_original['ID'].replace("id", "", regex=True, inplace=True) 

    # Start extending the dataset with "Signal_X_of_ID_Y" format
    df_extended = pd.DataFrame([])
    # Starting with 'ID','Label','Time' data from the original file
    df_extended[['ID','Label','Time']] = df_original[['ID','Label','Time']]
    
    # Repeating for all the unique CAN IDs
    for target_id in tqdm(df_original['ID'].unique()):
        # print("Targeted ID: ", target_id, end = ' ')
        # selecting the rows only with the a specific CAN ID and valid signals
        df_perID = df_original[df_original['ID'] == target_id].T.dropna().T

        # Appending ID number at the end of the signal column
        column_rename_dict = {}
        for org_column_name in set(df_perID.columns) - set(['ID','Label','Time']):
            new_column_name = org_column_name.replace("Signal", "Sig_")
            if "_of_ID" not in new_column_name:
                new_column_name = new_column_name + "_of_ID"
            new_column_name = new_column_name + "_" + str(target_id)
            # print(org_column_name, new_column_name)
            column_rename_dict[org_column_name] = new_column_name
        # renaming the columns
        df_perID = df_perID.rename(column_rename_dict, axis=1)
        new_column_names = list(column_rename_dict.values())
        
        #merging ID wise data frame in a singel combined one
        df_extended = pd.concat([df_extended, df_perID[new_column_names]], axis = 1)

    # Saving the dataset for future use
    extd_file_dir.parent.mkdir(parents=True, exist_ok=True)
    df_extended.to_csv(extd_file_dir, header = True, index = True)
    print(f"File saved to {extd_file_dir}")
    return df_extended

def get_minmax_scaler(features, dataset_name, scaler_dir):
    df_min_max = pd.read_csv(f"{scaler_dir}/min_max_values_{dataset_name}.csv"
                             , index_col=0)[features]
    scaler = MinMaxScaler()
    scaler.fit(df_min_max.values)
    print("scaler loaded...!")
    return scaler

def get_list_of_files(args): #data_type: str, clean_data_dir: str):
    data_type = args.data_type
    data_dir = args.data_dir

    file_dir_dict = {}
    file_paths = glob.glob(f"{data_dir}/*.csv")
    for file_path in file_paths:
        file_name = file_path.split("/")[-1].split(".")[0]
        file_dir_dict[file_name] = file_path
    file_dir_dict = OrderedDict(sorted(file_dir_dict.items()))
    return file_dir_dict
       
def load_data(dataset_name, file_name, file_path, features, org_features, dataset_fraction):
    
    # Load dataset
    file_path = Path(file_path)
    generated_data = Path(f"{file_path.parent}/generated/{file_name}_generated.csv")

    if generated_data.exists():
        print(f"Loading {generated_data}...")
        X = pd.read_csv(generated_data, index_col=0)
    else:
        print(f"{generated_data} does not exists!")
        X = generate_dataset(file_name, file_path, org_features)


    print(f"{file_name} loaded..")
    # Defining the number of samples 
    num_of_samples = int(X.shape[0]*dataset_fraction)
    y = X['Label'].iloc[0:num_of_samples]
    X = X[features].iloc[:num_of_samples].astype(float)
    print("Forward filling...")
    X = X.ffill().copy()
    X = X.bfill().dropna()   
    print("X_train.shape", X.shape)
    print("Done data treatment..")
    return X, y

def scale_dataset(X, dataset_name, features, scaler_dir):
    columns = X.columns
    X = X.values.copy()  
    scaler_train = get_minmax_scaler(features, dataset_name, scaler_dir)
    X = scaler_train.transform(X)
    X = pd.DataFrame(X, columns = columns)
    print("Dataset scalled!")
    return X

def load_scale_data(args, file_name, file_path):
    dataset_name = args.dataset_name
    org_features = args.org_features
    features = args.features
    dataset_fraction = args.dataset_fraction
    scaler_dir = args.scaler_dir
    X, y = load_data(dataset_name, file_name, file_path, features, org_features, dataset_fraction)
    X = scale_dataset(X, dataset_name, features, scaler_dir)
    return X, y