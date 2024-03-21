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

    att_columns = args.attributes

    X_train_all = pd.DataFrame([])
    X_test_all = pd.DataFrame([])
    var_df = pd.DataFrame([])

    for feat_domain in tqdm(args.domain_list):
        for windsize in tqdm(args.windsizes):
            data_type = 'training'
            f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feat_domain}_{windsize}.csv")
            x_train_feat  = pd.read_csv(f_name, index_col=0)
            f_name = ensure_dir(f"{args.results_dir}/variance_mapping_matrix_{dataset_name}_{data_type}_{feat_domain}_{windsize}.csv")
            var_feat = pd.read_csv(f_name, index_col=0)
            y_train_att = x_train_feat[att_columns]
            y_train_att['Label'] = 0
            x_train_feat = x_train_feat.drop(columns=att_columns)
            X_train_all = pd.concat([X_train_all, x_train_feat], axis=1)
            var_df = pd.concat([var_df, var_feat], axis=0)

            data_type = 'testing'
            f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{dataset_name}_{data_type}_{feat_domain}_{windsize}.csv")
            x_test_feat  = pd.read_csv(f_name, index_col=0)
            y_test_att = x_test_feat[att_columns]
            y_test_att['Label'] = y_test_att['Label'].astype(int)
            x_test_feat = x_test_feat.drop(columns=att_columns)
            X_test_all = pd.concat([X_test_all, x_test_feat], axis=1)

    var_th_list = args.varianceThresholds
    eval_data = []

    for feat_domain in args.domain_list + ['all']:
        for var_th in var_th_list:
            high_variance_map = var_df.copy()
            high_variance_map_index = (high_variance_map[args.features] > var_th).astype(int).values
            high_variance_map.loc[:, args.features] = high_variance_map_index
            if feat_domain != 'all':
                feat_map_per_domain = high_variance_map[high_variance_map['Domain'] == feat_domain].copy()
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
                plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset_name}/loss_plot_rolling_{dataset_name}_{feat_domain}_{windsize}_{var_th}.jpg"), dpi=500)
                plt.show()
                scale_plot_loss(args, X_test_loss, y_test_att, model_name, feat_domain, windsize)
                eval_data += calc_roc_auc(args, windsize, 'Loss', X_test_loss, y_test_att, dataset_name, feat_domain, var_th, num_of_final_feat, min_var, model_name)

    eval_columns = ['Dataset', 'Domain', 'VarianceTh', 'NumOfFeat', 'Min_Var_PCA', 'Model', 'Attack', 'AUC']
    eval_df = pd.DataFrame(eval_data, columns=eval_columns)
    eval_df.to_csv(f"{args.results_dir}/eval_df_{dataset_name}.csv", header=True, index=True)

if __name__ == "__main__":
    feature_analysis()