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

import json
from os.path import exists as file_exists
from pathlib import Path
from typing import Dict

import pandas as pd
from dataset.load_dataset import *
from helpers import *

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def ensure_dir(file_directory):
    """
    Ensure the directory exists. If not, create it.

    Parameters:
        file_directory (str): Path to the directory.

    Returns:
        Path: Path object representing the directory.
    """
    file_directory = Path(file_directory)
    if not file_directory.parent.exists():
        file_directory.parent.mkdir(parents=True, exist_ok=True)
        print("Parent directory created!")
    return file_directory

def save_signal_map(args):
    """
    Save signal mappings to CSV and JSON files.

    Parameters:
        args (object): An object containing necessary arguments including:
            - results_dir (str): Directory to save the results.
            - dataset_name (str): Name of the dataset.
            - features (list): List of features or signals.

    Returns:
        dict: Signal mapping dictionary.
    """
    try:
        # Try to read existing mapping files
        csv_file_path = f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv"
        signal_mapping_df = pd.read_csv(csv_file_path, index_col=0)
        
        json_file_path = f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt'
        with open(json_file_path, 'r') as json_file:
            signal_mapping = json.load(json_file)
    except FileNotFoundError:
        # If mapping files don't exist, create and save them
        list_of_signals_dict = {args.dataset_name: args.features}
        
        signal_mapping_dict = {}
        for dataset_name, list_of_signals_dataset in list_of_signals_dict.items():
            signal_mapping = {}
            for id, signal in enumerate(list_of_signals_dataset):
                signal_mapping[signal] = id
                
            # Save mapping to JSON file
            json_file_path = ensure_dir(f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt')
            with open(json_file_path, 'w') as fp:
                fp.write(json.dumps(signal_mapping))
            
            # Convert mapping to DataFrame and save to CSV
            signal_mapping_df = pd.DataFrame(pd.Series(signal_mapping), columns=['Index'])
            signal_mapping_df['Signal'] = signal_mapping_df.index
            csv_file_path = ensure_dir(f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv")
            signal_mapping_df.to_csv(csv_file_path, header=True, index=True)
    return signal_mapping

def fit_test_pca(X_train, X_test, num_pr_comps):
    """
    Perform Principal Component Analysis (PCA) on the data.

    Parameters:
        X_train (DataFrame): Training data.
        X_test (DataFrame): Test data.
        num_pr_comps (int): Number of principal components.

    Returns:
        DataFrame: Reconstruction loss.
    """
    print("Starting PCA...")
    
    # Principal Component Analysis
    pca = PCA(random_state=22, n_components=num_pr_comps)
    pca.fit(X_train.values)

    # Transforming test data
    X_test_pc = pd.DataFrame(pca.transform(X_test.values))
    
    # Reconstructing original features
    X_test_recon = pd.DataFrame(pca.inverse_transform(X_test_pc.values), columns=X_test.columns)
    
    # Reconstruction loss after PCA
    X_test_loss = abs(X_test - X_test_recon)
    return X_test_loss


def scale_plot_loss(args, X_test_loss, y_test_att, model_name, domain, windsize):
    """
    Scale and plot loss values.

    Parameters:
        args (object): An object containing necessary arguments including:
            - plot_dir (str): Directory to save the plots.
            - dataset_name (str): Name of the dataset.
        X_test_loss (DataFrame): Loss values.
        y_test_att (DataFrame): Test data labels.
        model_name (str): Name of the model.
        domain (str): Domain of the features.
        windsize (int): Window size.

    Returns:
        None
    """
    dataset = args.dataset_name
    
    scaler_loss = MinMaxScaler()
    X_test_loss_sc = pd.DataFrame(scaler_loss.fit_transform(X_test_loss), columns=X_test_loss.columns)
    
    plt.figure()
    X_test_loss_sc.iloc[:1000, 0:].plot()
    y_test_att['Label'][0:1000].plot()
    plt.legend([])
    plt.figure()
    X_test_loss_sc.mean(axis=1).rolling(1).mean().fillna(0)[:].plot()
    y_test_att['Label'].plot()
    plt.legend([])
    plt.savefig(f"{args.plot_dir}/{dataset}/anom_score_{dataset}_{model_name}_{domain}_{windsize}.jpg", dpi=500)
    # plt.show()
    plt.close()

def calc_roc_auc(args, windsize, input_type, X_test_loss_pred, y_test_att, dataset, domain, var_th, num_of_final_feat, min_var, model_name):
    """
    Calculate AUROC and plot ROC curve.

    Parameters:
        args (object): An object containing necessary arguments including:
            - plot_dir (str): Directory to save the plots.
        windsize (int): Window size.
        input_type (str): Type of input (Loss or Pred).
        X_test_loss_pred (DataFrame): Predicted loss values.
        y_test_att (DataFrame): Test data labels.
        dataset (str): Name of the dataset.
        domain (str): Domain of the features.
        var_th: Threshold for variance.
        num_of_final_feat: Number of final features.
        min_var: Minimum variance.
        model_name (str): Name of the model.

    Returns:
        list: Evaluation data.
    """
    roll_win = 1
    plt.figure(figsize=(7, 4))
    eval_data = []

    for file_name in sorted(y_test_att['File'].unique()):
        if 'normal' in file_name:
            continue
        file_indices = y_test_att['File'] == file_name
        if input_type == 'Loss':
            y_score = X_test_loss_pred.loc[file_indices].mean(axis=1).rolling(roll_win).mean().fillna(0)
        elif input_type == 'Pred':
            y_score = X_test_loss_pred.loc[file_indices].rolling(roll_win).mean().fillna(0)
        y_test = y_test_att['Label'].loc[file_indices].values
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        attack_short_name = file_name
        plt.plot(fpr, tpr, label=f"{attack_short_name.capitalize()}, AUC: {round(roc_auc, 3)}")
        eval_data.append([dataset, domain, var_th, num_of_final_feat, min_var, model_name, file_name, round(roc_auc, 3)])
        
    plt.legend()
    plt.title(f"ROC Curve using {domain} features with {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{args.plot_dir}/{dataset}/ROC_AUC_{dataset}_{model_name}_{domain}_{windsize}_{var_th}.jpg", dpi=500)
    # plt.show()
    plt.close()

    return eval_data
