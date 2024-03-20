import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists as file_exists
from pathlib import Path
from typing import Dict
import joblib
from sklearn.preprocessing import MinMaxScaler
import tsfel
import hydra
from dataset.load_dataset import *
from helper import *

from sklearn.decomposition import PCA
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def save_signal_map(args):
    try:
        f_name = ensure_dir(f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv")
        signal_mapping_df  = pd.read_csv(f_name,  index_col=0)
        f_name = f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt'
        with open(f_name,'r') as json_file:
            signal_mapping = json.load(json_file)
    except:
        print("Going to except...")
        list_of_signals_dict = {}
        list_of_signals_dict[args.dataset_name] = args.features
        #signal-id mapping dict
        signal_mapping_dict = {}
        for dataset_name, list_of_signals_dataset in list_of_signals_dict.items():
            signal_mapping = {}
            for id, signal in enumerate(list_of_signals_dataset):
                signal_mapping[signal] = id
            f_name = ensure_dir(f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt')
            with open(f_name,'w') as fp:
                fp.write(json.dumps(signal_mapping))
            signal_mapping_df = pd.DataFrame(pd.Series(signal_mapping), columns = ['Index'])
            signal_mapping_df['Signal'] = signal_mapping_df.index
            f_name = ensure_dir(f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv")
            signal_mapping_df.to_csv(f_name, header = True, index=True)
    return signal_mapping

def ensure_dir(file_directory):
    file_directory = Path(file_directory)
    if not file_directory.parent.exists():
        file_directory.parent.mkdir(parents=True, exist_ok=True)
        print("Parent directory created!")
    return file_directory

def fit_test_pca(X_train, X_test, num_pr_comps):
    '''
    arg: 
    X: data
    num_pr_comps: number of principal components

    return:
    x_loss: reconstruction loss

    '''

    # -------------------- Principal Component Analysis----------------------
    print("Starting PCA...")
    #--------- Train data-------------------------------
    # Defining and fitting PCA model...
    pca = PCA(random_state=22, n_components=num_pr_comps)
    pca.fit(X_train.values)

    #--------- Test data-------------------------------
    # Extracting principal components of the test data...
    X_test_pc = pd.DataFrame(pca.transform(X_test.values))
    # reconstructing original features...
    X_test_recon = pd.DataFrame(pca.inverse_transform(X_test_pc.values), columns = X_test.columns)
    # reconstruction loss after PCA...
    X_test_loss = abs(X_test - X_test_recon)
    return X_test_loss
    #--------------------------------------------------------------------------

def scale_plot_loss(args, X_test_loss, y_test_att, model_name,  domain, windsize):
    # --------------- Scale and Plot Loss -------------------------------------
    # Scaling loss values from 0 to 1 for further processing...

    dataset = args.dataset_name
    
    scaler_loss = MinMaxScaler()
    X_test_loss_sc = pd.DataFrame(scaler_loss.fit_transform(X_test_loss), columns = X_test_loss.columns)
    # plotting reconstruction loss for different features...
    plt.figure()
    X_test_loss_sc.iloc[:1000,0:].plot()
    y_test_att['Label'][0:1000].plot()
    plt.legend([])
    #--------------------------------------------------------------------------
    plt.figure()
    X_test_loss_sc.mean(axis = 1).rolling(1).mean().fillna(0)[:].plot()
    y_test_att['Label'].plot()
    plt.legend([])
    plt.savefig(f"{args.plot_dir}/{dataset}/anom_score_{dataset}_{model_name}_{domain}_{windsize}.jpg", dpi = 500)
    plt.show()
    #--------------------------------------------------------------------------
    
def calc_roc_auc(args, windsize, input_type, X_test_loss_pred, y_test_att, dataset, domain, var_th, num_of_final_feat, min_var, model_name):
        
    # --------------- Calculate AUROC and Plot ROC Curve --------------------------------
    # Define the plots...
    #TODO: Evaluate Rolling Window and Mean/Max Operation...
    roll_win = 1

    plt.figure(figsize=(7,4))
    eval_data = []

    for file_name in sorted(y_test_att['File'].unique()): 
            
        print(file_name)
        
        if 'normal' in file_name:
            continue

        # Adding filter per file name...
        file_indices = y_test_att['File'] == file_name
        
        # Anomaly Score....
        if input_type == 'Loss':
            y_score = X_test_loss_pred.loc[file_indices].mean(axis = 1).rolling(roll_win).mean().fillna(0)
        elif input_type =='Pred':
            y_score = X_test_loss_pred.loc[file_indices].rolling(roll_win).mean().fillna(0)

        # Ground truth...
        y_test = y_test_att['Label'].loc[file_indices].values
        #%----------------------------------
        y_check = list(np.where(y_test == 1)[0])
        
        # Filling missing points where there was no attacked messges..    
        if dataset == 'road':
            try:
                start = y_check[0]
                end = y_check[-1]
                y_test[start:end] = 1
            except:
                pass
        #-----------------------------------
        # Find True Positive Rate, and False Positive Rate & AUC score
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Attack Rename...
        # attack_short_name = file_name.split("_")[1]
        attack_short_name = file_name
        plt.plot(fpr, tpr,  label = f"{attack_short_name. capitalize()}, AUC: {round(roc_auc,3)}") #marker = '.',
        #-----------------------------------
        eval_data.append([dataset, domain, var_th, num_of_final_feat, min_var, model_name, file_name, round(roc_auc,3)])
        #-----------------------------------
        
    plt.legend()
    plt.title(f"ROC Curve using {domain} features with {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{args.plot_dir}/{dataset}//ROC_AUC_{dataset}_{model_name}_{domain}_{windsize}_{var_th}.jpg", dpi = 500)
    plt.show()

    return eval_data
    #----------------
