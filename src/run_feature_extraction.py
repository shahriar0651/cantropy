import json
from os.path import exists as file_exists
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsfel
import hydra
from tqdm import tqdm

# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
from dataset.load_dataset import get_list_of_files, load_scale_data
from training import *

# Utility function to ensure directory existence
def ensure_dir(file_directory):
    file_directory = Path(file_directory)
    if not file_directory.parent.exists():
        file_directory.parent.mkdir(parents=True, exist_ok=True)
        print("Parent directory created!")
    return file_directory

# Function to save signal mapping
def save_signal_map(args):
    try:
        f_name = ensure_dir(f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv")
        signal_mapping_df = pd.read_csv(f_name, index_col=0)
        f_name = f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt'
        with open(f_name, 'r') as json_file:
            signal_mapping = json.load(json_file)
    except:
        print("Going to except...")
        list_of_signals_dict = {args.dataset_name: args.features}
        signal_mapping_dict = {}
        for dataset_name, list_of_signals_dataset in list_of_signals_dict.items():
            signal_mapping = {signal: id for id, signal in enumerate(list_of_signals_dataset)}
            f_name = ensure_dir(f'{args.results_dir}/signal_mapping_{args.dataset_name}.txt')
            with open(f_name, 'w') as fp:
                json.dump(signal_mapping, fp)
            signal_mapping_df = pd.DataFrame(pd.Series(signal_mapping), columns=['Index'])
            signal_mapping_df['Signal'] = signal_mapping_df.index
            f_name = ensure_dir(f"{args.results_dir}/signal_mapping_{args.dataset_name}.csv")
            signal_mapping_df.to_csv(f_name, header=True, index=True)
    return signal_mapping

# Main function for feature extraction
@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def feature_extraction(args: dict) -> None:
    # Get the root directory
    root_dir = Path(__file__).resolve().parent
    args.root_dir = root_dir

    # Set data directory based on data type
    data_type = args.data_type
    args.data_dir = args.train_data_dir if data_type == "training" else args.test_data_dir
    features = args.features
    dataset = args.dataset_name

    # Initialize dictionary to store feature names
    feats_dict = {}
    # Initialize dataframe to store final windows
    df_windows_final = pd.DataFrame([])
    
    # Save signal mapping
    signal_mapping = save_signal_map(args)

    # Get file directory dictionary
    file_dir_dict = get_list_of_files(args)

    load = False
    generate = True

    # Process each file
    for file_name, file_path in tqdm(file_dir_dict.items(), desc="Files processed"):
        windsizes_missing_dict = {feature_type: [] for feature_type in args.feature_type_list}
        df_windows_file = pd.DataFrame([])

        # Process each feature type and windsize
        for feature_type in args.feature_type_list:
            for windsize in args.windsizes:
                f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_{args.dataset_name}_{file_name}_{args.per_of_samples}_{feature_type}_{windsize}.csv")

                if file_exists(f_name):
                    if not load:
                        continue
                    df_windows_feature = pd.read_csv(f_name, index_col=0)
                    df_windows_file = pd.concat([df_windows_file, df_windows_feature], axis=1, ignore_index=False)
                    feats_dict[feature_type] = df_windows_feature.columns
                    windsizes_missing_dict[feature_type].append(0)
                else:
                    if not generate:
                        continue
                    windsizes_missing_dict[feature_type].append(windsize)

        df_windows_final = pd.concat([df_windows_final, df_windows_file], axis=0, ignore_index=True)

        # Check if all features are extracted
        if sum(sum(v) for v in windsizes_missing_dict.values()) == 0:
            print(f"All the features are extracted from {file_name}")
        else:
            print(f"Loading the features for windows", windsizes_missing_dict)
            print("Loading dataset:", file_name)

            try:
                X_train, y_train = load_scale_data(args, file_name, file_path)
            except Exception as error:
                print(error)
                print(f"Skipping dataset {file_name}")

            # Extract features for missing windsizes
            for feature_type in args.feature_type_list:
                cgf_file = tsfel.get_features_by_domain(feature_type)
                windsizes_missing = windsizes_missing_dict[feature_type].copy()

                for windsize in windsizes_missing:
                    df_windows_file = pd.DataFrame([])
                    total_batch = int(len(X_train) / args.batch_size) + 1

                    for i in range(total_batch):
                        y_cut = y_train[i * args.batch_size:(i + 1) * args.batch_size].copy()
                        df_windows = tsfel.time_series_features_extractor(cgf_file, X_train[i * args.batch_size:(i + 1) * args.batch_size], window_size=windsize, fs=50, resample_rate=100, n_jobs=-1)
                        df_windows['File'] = file_name
                        df_windows['Feature'] = feature_type
                        df_windows['Window'] = windsize
                        df_windows['Label'] = y_cut.rolling(windsize).max().dropna().iloc[::windsize].values

                        df_windows_file = pd.concat([df_windows_file, df_windows], ignore_index=True)

                    f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_{args.dataset_name}_{file_name}_{args.per_of_samples}_{feature_type}_{windsize}.csv")
                    df_windows_file.to_csv(f_name, index=True, header=True)

    # Combine all data
    load = True
    generate = False

    for feature_type in args.feature_type_list:
        df_windows_final = pd.DataFrame([])

        for file_name, file_path in tqdm(file_dir_dict.items(), desc="Files processed"):
            df_windows_file = pd.DataFrame([])

            for windsize in args.windsizes:
                f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_{args.dataset_name}_{file_name}_{args.per_of_samples}_{feature_type}_{windsize}.csv")

                if file_exists(f_name):
                    if load:
                        df_windows_feature = pd.read_csv(f_name, index_col=0)
                        df_windows_file = pd.concat([df_windows_file, df_windows_feature], axis=1, ignore_index=False)

            df_windows_final = pd.concat([df_windows_final, df_windows_file], axis=0, ignore_index=True)

        feat_columns = sorted(list(set(df_windows_final.columns) - {'File', 'Feature', 'Window', 'Label'}))
        df_windows_feats = df_windows_final[['File', 'Feature', 'Window', 'Label']].copy()

        # Scale features
        scaler_filename = f"{args.scaler_dir}//scaler_data_{args.dataset_name}_{feature_type}_{windsize}.save"

        if data_type == 'training':
            scaler_data = MinMaxScaler()
            df_windows_final = scaler_data.fit_transform(df_windows_final[feat_columns].values)
            joblib.dump(scaler_data, scaler_filename)
        else:
            scaler_data = joblib.load(scaler_filename)
            df_windows_final = scaler_data.transform(df_windows_final[feat_columns].values)

        df_windows_final = pd.DataFrame(df_windows_final, columns=feat_columns)
        df_windows_final = pd.concat([df_windows_final, df_windows_feats], axis=1)
        
        # Taking care of Nan values
        df_windows_final = df_windows_final.ffill()
        df_windows_final = df_windows_final.bfill()
        df_windows_final = df_windows_final.fillna(0.0)
        #---------------------------
        f_name = ensure_dir(f"{args.features_dir}/df_windows_tsfel_clean_{args.dataset_name}_{data_type}_{feature_type}_{windsize}.csv")
        df_windows_final.to_csv(f_name, index=True, header=True)
        print(f_name, "saved!!!")

        # Analyze variance
        # if data_type == 'testing':
        #     return None
        
        # Get variance of each feature
        non_string_features = df_windows_final.select_dtypes(exclude=['object']).columns.tolist()
        df_windows_var = df_windows_final[non_string_features].var()

        # Create unique features list
        float_selected = list(set(df_windows_final.columns[df_windows_final.dtypes == float]))
        float_selected.remove("Label")
        list_of_unique_feat = list(set(["_".join(x.split("_")[5:]) for x in list(float_selected)]))

        # Create feature dictionary for panda dataframe
        feat_dict = {}
        cgf_file = tsfel.get_features_by_domain() 

        for domain, domain_feat in cgf_file.items():
            for feature, feat_config in domain_feat.items():
                count = 1
                for unique_feat in list_of_unique_feat:
                    if feature in unique_feat:
                        feat_dict[unique_feat] = {'Domain': domain, 'Feature': feature, 'Feature no': count}
                        count += 1
                        
        feature_df = pd.DataFrame(feat_dict).T.sort_values(by=['Domain', 'Feature'], ascending=True)

        # Add variance data to feature dataframe
        for signal_name in features:
            list_of_features = [f"{signal_name}_{feat}" for feat in list_of_unique_feat]
            df_var = pd.DataFrame(df_windows_var[list_of_features].values, index=list_of_unique_feat, columns=[signal_name])
            feature_df = pd.concat([feature_df, df_var], axis=1, ignore_index=False)   

        # Plot variance
        var_df = feature_df.groupby(['Feature']).max()[features]
        var_df = var_df.T[var_df.median(axis=1).sort_values().index.to_list()]

        plt.figure(figsize=(12, 2.5))
        boxplot = var_df.boxplot()  
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_box_plot_{data_type}_{dataset}_{feature_type}.jpg"))
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_box_plot_{data_type}_{dataset}_{feature_type}.pdf"))
        plt.show()

        plt.figure(figsize=(12, 2.5))
        var_df.T.plot(linestyle='--', marker='p', markersize='3', figsize=(12, 1.5))
        plt.legend([])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_line_plot_{data_type}_{dataset}_{feature_type}.jpg"))
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_line_plot_{data_type}_{dataset}_{feature_type}.pdf"))
        plt.show()

        # List of features with different variance threshold
        var_list = np.arange(0, 30, 2)/100
        count_df = pd.DataFrame([])
        for var in var_list:
            var_df_bin = (var_df >= var).astype(int)
            count_var = pd.DataFrame(var_df_bin.sum(), columns=['Feature-wise Count'])
            count_var['Feature'] = count_var.index
            count_var['Variance Threshold'] = var
            count_df = pd.concat([count_df, count_var], axis=0, ignore_index=True)

        count_df_total = count_df.groupby(['Variance Threshold']).sum()
        count_df_total['Variance Threshold'] = count_df_total.index

        count_df['Feature-wise Count'] = count_df['Feature-wise Count']/len(features)*100
        count_df_total['Feature-wise Count'] = count_df_total['Feature-wise Count']

        # Plot variance vs. feature count distribution
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        sns.lineplot(count_df_total, x='Variance Threshold', y='Feature-wise Count', linestyle='--', linewidth='0.5', marker='p', markersize='10', ax=ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_vs_feature_count_dist_{data_type}_{dataset}_{feature_type}.jpg"))
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/variance_vs_feature_count_dist_{data_type}_{dataset}_{feature_type}.pdf"))
        plt.show()

        # Plot boxplot variance vs. feature count distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.boxplot(count_df, x='Variance Threshold', y='Feature-wise Count', color='gray', ax=ax)
        sns.stripplot(count_df, x='Variance Threshold', y='Feature-wise Count', hue='Feature', ax=ax)
        plt.legend(bbox_to_anchor=(1, 1), ncols=3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/boxplot_variance_vs_feature_count_dist_{data_type}_{dataset}_{feature_type}.jpg"))
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/boxplot_variance_vs_feature_count_dist_{data_type}_{dataset}_{feature_type}.pdf"))
        plt.show()

        # Plot lineplot variance vs. feature count distribution
        sns.lineplot(count_df, x='Variance Threshold', y='Feature-wise Count', hue='Feature', linestyle='--', linewidth='0.5', marker='p', markersize='10')
        plt.legend(bbox_to_anchor=(1, 1.05), ncols=3)
        plt.tight_layout()
        plt.savefig(ensure_dir(f"{args.plot_dir}/{dataset}/lineplot_variance_vs_feature_count_dist_{data_type}_{dataset}_{feature_type}.jpg"))
        plt.show()

        # Save variance data
        f_name = ensure_dir(f"{args.results_dir}/feature_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv")
        feature_df.to_csv(f_name, header=True, index=True)

        f_name = ensure_dir(f"{args.results_dir}/variance_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv")
        var_df.to_csv(f_name, header=True, index=True)

if __name__ == "__main__":
    feature_extraction()

