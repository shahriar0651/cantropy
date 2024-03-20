import numpy as np
import pandas as pd
import joblib
import hydra
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from dataset import *
# from helper import save_signal_map, get_list_of_files, ensure_dir, scale_plot_loss, calc_roc_auc, fit_test_pca
from helpers import *

@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def feature_analysis(args: dict) -> None:
    root_dir = Path(__file__).resolve().parent
    args.root_dir = root_dir
    args.data_dir = args.train_data_dir if args.data_type == "training" else args.test_data_dir
    dataset_name = args.dataset_name

    feats_dict = {}
    df_windows_final = pd.DataFrame([])
    signal_mapping = save_signal_map(args)
    file_dir_dict = get_list_of_files(args)

    load = True
    generate = False
    att_columns = ['File','Feature','Window', 'Label']

    X_train_all = pd.DataFrame([])
    X_test_all = pd.DataFrame([])
    var_df = pd.DataFrame([])

    for feature_type in tqdm(args.feature_type_list):
        for windsize in tqdm(args.windsizes):
            data_type = 'training'
            f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
            x_train_feat  = pd.read_csv(f_name, index_col=0)
            f_name = ensure_dir(f"{args.results_dir}/feature_matrix_{dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
            var_feat = pd.read_csv(f_name, index_col=0)
            y_train_att = x_train_feat[att_columns]
            y_train_att['Label'] = 0
            x_train_feat = x_train_feat.drop(columns=att_columns)
            X_train_all = pd.concat([X_train_all, x_train_feat], axis=1)
            var_df = pd.concat([var_df, var_feat], axis=0)

            data_type = 'testing'
            f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
            x_test_feat  = pd.read_csv(f_name, index_col=0)
            y_test_att = x_test_feat[att_columns]
            y_test_att['Label'] = y_test_att['Label'].astype(int)
            x_test_feat = x_test_feat.drop(columns=att_columns)
            X_test_all = pd.concat([X_test_all, x_test_feat], axis=1)

    print(X_train_all.shape, ">>>>", X_train_all.dropna().shape)
    print(X_test_all.shape, ">>>>", X_test_all.dropna().shape)

    var_th_list = [0.0, 0.005, 0.01, 0.015, 0.02]
    eval_data = []

    for domain in args.feature_type_list + ['all']:
        for var_th in var_th_list:
            high_variance_map = var_df.copy()
            high_variance_map_index = (high_variance_map[args.features] > var_th).astype(int).values
            high_variance_map.loc[:, args.features] = high_variance_map_index
            if domain != 'all':
                feat_map_per_domain = high_variance_map[high_variance_map['Domain'] == domain].copy()
            else:
                feat_map_per_domain = high_variance_map.copy()

            selected_features = []
            for signal in args.features:
                feat_map_per_signal = feat_map_per_domain.index[feat_map_per_domain[signal].astype(bool)]
                selected_features += [f"{signal}_{feat_sig}" for feat_sig in feat_map_per_signal]

            print(f"Selected features after filtering: {X_train_all.shape[1]} >> {len(selected_features)}")

            X_train = X_train_all[selected_features].copy()
            X_test = X_test_all[selected_features].copy()

            num_of_final_feat = X_test.shape[1]

            pca = PCA(random_state=22)
            pca.fit(X_train)
            for min_var in [99.99999]:
                var = []
                for i in range(len(pca.explained_variance_ratio_)):
                    var_sum = np.sum(pca.explained_variance_ratio_[0:i])
                    var.append(var_sum)
                    if var_sum >= min_var / 100:
                        break
                num_pr_comps = len(var)
                print("Number of principal components: ", num_pr_comps)
                model_name = "PCA"
                X_test_loss = fit_test_pca(X_train, X_test, num_pr_comps)
                scaler_loss = MinMaxScaler()
                X_test_loss_sc = pd.DataFrame(scaler_loss.fit_transform(X_test_loss), columns=X_test_loss.columns)
                plt.figure()
                X_test_loss_sc.mean(axis=1).rolling(2).mean().fillna(0)[:].plot()
                y_test_att['Label'].plot()
                plt.legend([])
                plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset_name}/loss_plot_rolling_{dataset_name}_{domain}_{windsize}_{var_th}.jpg"), dpi=500)
                plt.show()
                scale_plot_loss(args, X_test_loss, y_test_att, model_name, domain, windsize)
                eval_data += calc_roc_auc(args, windsize, 'Loss', X_test_loss, y_test_att, dataset_name, domain, var_th, num_of_final_feat, min_var, model_name)

    eval_columns = ['Dataset', 'Domain', 'VarianceTh', 'NumOfFeat', 'Min_Var_PCA', 'Model', 'Attack', 'AUC']
    eval_df = pd.DataFrame(eval_data, columns=eval_columns)
    eval_df.to_csv(f"{args.results_dir}/eval_df_{dataset_name}.csv", header=True, index=True)

if __name__ == "__main__":
    feature_analysis()


# from tqdm import tqdm
# from os.path import exists as file_exists
# from pathlib import Path
# from typing import Dict
# import joblib
# from sklearn.preprocessing import MinMaxScaler
# import tsfel
# import hydra
# from dataset.load_dataset import *
# from training import *
# from helper import *

# from sklearn.decomposition import PCA
# import pandas as pd

# from sklearn.preprocessing import MinMaxScaler 
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.metrics import roc_curve, auc


# @hydra.main(version_base=None, config_path="../config", config_name="syncan")
# def feature_analysis(args: dict) -> None:
#     # Get the root directory
#     root_dir = Path(__file__).resolve().parent
#     args.root_dir = root_dir
    
#     # Set data directory based on data type
#     data_type = args.data_type
#     args.data_dir = args.train_data_dir if data_type == "training" else args.test_data_dir
#     dataset_name = args.dataset_name
#     dataset = args.dataset_name
#     features = args.features
#     # Initialize dictionary to store feature names
#     feats_dict = {}
#     # Initialize dataframe to store final windows
#     df_windows_final = pd.DataFrame([])

#     signal_mapping = save_signal_map(args)

#     # Get file directory dictionary
#     file_dir_dict = get_list_of_files(args)

#     load = True
#     generate = False

#     att_columns = ['File','Feature','Window', 'Label'] #FIXME : copy to yaml

#     # Dataframe to save all the extracted features...
#     X_train_all = pd.DataFrame([])
#     X_test_all = pd.DataFrame([])
#     var_df = pd.DataFrame([])


#     # Looping over the features and windowsizes...
#     data_type = 'training'
#     for feature_type in tqdm(args.feature_type_list):
#         for windsize in tqdm(args.windsizes):
#             # -------------------- Training-----------------
#             # Rading csv files generated in the feature extraction process...
#             data_type = 'training'
#             f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
#             x_train_feat  = pd.read_csv(f_name, index_col = 0)
#             f_name = ensure_dir(f"{args.results_dir}/feature_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv")
#             var_feat = pd.read_csv(f_name, index_col = 0) #FIXME : Feature extraction

#             # f_name = ensure_dir(f"{args.results_dir}/variance_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv")
#             # var_df.to_csv(f_name, header=True, index=True)

#             y_train_att = x_train_feat[att_columns]
#             y_train_att['Label'] = 0

#             x_train_feat = x_train_feat.drop(columns=att_columns)
#             X_train_all = pd.concat([X_train_all, x_train_feat], axis = 1)
#             print("x_train_feat: ", x_train_feat)

#             var_df = pd.concat([var_df, var_feat], axis= 0) #FIXME : Feature extraction

#             print("Training: ", feature_type, X_train_all.shape, x_train_feat.shape)


#             # -------------------- Test -----------------
#             # Rading csv files generated in the feature extraction process...
#             data_type = 'testing'
#             # Rading csv files generated in the feature extraction process...
#             f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
#             x_test_feat  = pd.read_csv(f_name, index_col = 0)
                    
#             y_test_att = x_test_feat[att_columns]
#             y_test_att['Label'] = y_test_att['Label'].astype(int) 
#             x_test_feat = x_test_feat.drop(columns=att_columns)
#             print("x_test_feat: ", x_test_feat)

#             X_test_all = pd.concat([X_test_all, x_test_feat], axis = 1)
#             print("Testing: ", feature_type, X_test_all.shape, x_test_feat.shape)
    
#     print(X_train_all.shape, ">>>>", X_train_all.dropna().shape)
#     print(X_test_all.shape, ">>>>", X_test_all.dropna().shape)

#     print(y_train_att['File'].value_counts())
#     # Starting loop for types of features/domain
#     var_th_list = [0.0, 0.005, 0.01, 0.015, 0.02] #, 0.05, 0.10]
#     eval_data = []


#     # Starting with individual domain................................................................
#     #=================================================================================================
#     for domain in args.feature_type_list +['all']:
#     # for domain in ['temporal']:
#         # Starting with different variance threshold....
#         for var_th in var_th_list:

#             # Selecting features with minimum variance in training data
#             high_variance_map = var_df.copy()
#             high_variance_map_index = (high_variance_map[features] > var_th).astype(int).values
#             high_variance_map.loc[:, features] = high_variance_map_index
#             # print((high_variance_map[features] > var_th).astype(int))

#             print(f"------------- Variance: {var_th}, Domain: {domain} -------------")
#             # Creating filter to consider features only from specific domain...
#             if domain != 'all':
#                 feat_map_per_domain = high_variance_map[high_variance_map['Domain'] == domain].copy()
#                 print(feat_map_per_domain.shape, high_variance_map.shape)
#             else:
#                 feat_map_per_domain = high_variance_map.copy()
#                 print(feat_map_per_domain.shape, high_variance_map.shape)


#             # Looping over individual signal to find the list of important features
#             selected_features = []
#             for signal in features:
#                 feat_map_per_signal = feat_map_per_domain.index[feat_map_per_domain[signal].astype(bool)]
#                 selected_features += [f"{signal}_{feat_sig}" for feat_sig in feat_map_per_signal]

#             print(f"Selected features after filtering: {X_train_all.shape[1]} >> {len(selected_features)}")

#             for further_merging in [True, False]:
#                 # Select the features from the overall dataset...
#                 X_train = X_train_all[selected_features].copy()
#                 X_test = X_test_all[selected_features].copy()

            
#                 if further_merging == True:
#                     # Getting the list of generic features (remove the last extension)
#                     generic_feat = []
#                     for col in X_train.columns.to_list():
#                         cols = col.split("_")
#                         generic_feat.append(f"{cols[0]}_{cols[1]}")
#                     # Removing generic features...
#                     X_train = X_train.T
#                     X_train['Feature'] = generic_feat
#                     X_train = X_train.groupby('Feature').max().T

#                     X_test = X_test.T
#                     X_test['Feature'] = generic_feat
#                     X_test = X_test.groupby('Feature').max().T
#                     print(f"Selected features after merging: {len(selected_features)} >> {X_test.shape[1]}")
                
#                 num_of_final_feat = X_test.shape[1]


#                 # Doing PCA to linearly analyze the dataset...................................................
#                 #==============================================================================================
#                 # fitting PCA model...
#                 pca = PCA(random_state=22)
#                 pca.fit(X_train)

#                 # Increasing the number of PCs to gain 99.9% variancee 
#                 # min_var = 0.999 # setting the var thresholds...
#                 for min_var in [99.99999]: #[99.0, 99.99, 99.999, 99.9999, 99.99999, 99.999999]:
                                
#                     var = []
#                     # -------------------------------------------
#                     for i in range(len(pca.explained_variance_ratio_)):
#                         var_sum = np.sum(pca.explained_variance_ratio_[0:i])
#                         var.append(var_sum)
#                         if var_sum >= min_var/100:
#                             break

#                     #-------------------------------------------
#                     # Defining number of components in the PCA...
#                     num_pr_comps = len(var)
#                     print("Number of princial components: ", num_pr_comps)


#                     model_name = "PCA"
#                     X_test_loss = fit_test_pca(X_train, X_test, num_pr_comps)

                    
#                     # --------------- Scale and Plot Loss -------------------------------------
#                     # Scaling loss values from 0 to 1 for further processing...
#                     scaler_loss = MinMaxScaler()
#                     X_test_loss_sc = pd.DataFrame(scaler_loss.fit_transform(X_test_loss), columns = X_test_loss.columns)
                    
#                     plt.figure()
#                     X_test_loss_sc.mean(axis = 1).rolling(2).mean().fillna(0)[:].plot()
#                     y_test_att['Label'].plot()
#                     plt.legend([])
#                     plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/loss_plot_rolling_{dataset}_{domain}_{windsize}_{var_th}_{further_merging}.jpg"), dpi = 500)
#                     plt.show()
#                     #--------------------------------------------------------------------------
                    
                
#                     scale_plot_loss(args, X_test_loss, y_test_att, model_name,  domain, windsize)
#                     eval_data += calc_roc_auc(args, windsize, 'Loss', X_test_loss, y_test_att, dataset, domain, var_th, num_of_final_feat, min_var, further_merging, model_name)   
#                     print(eval_data)
                    
                    
        
#         eval_columns = ['Dataset', 'Domain', 'VarianceTh', 'NumOfFeat', 'Min_Var_PCA', 'Merge', 'Model', 'Attack', 'AUC']
#         eval_df = pd.DataFrame(eval_data)
#         eval_df.columns = eval_columns
#         eval_df.to_csv(f"{args.results_dir}/eval_df_{dataset}.csv", header = True, index = True)
        
# if __name__ == "__main__":
#     feature_analysis()




