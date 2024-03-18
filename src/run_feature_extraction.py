from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from os.path import exists as file_exists    
from dataset.load_dataset import *
import tsfel
import joblib

# from hydra.utils import get_original_cwd
from training import *


@hydra.main(version_base=None, config_path="../config", config_name="syncan")
def develop_canshield(args : DictConfig) -> None:
    root_dir = Path(__file__).resolve().parent
    print("root_dir: ", root_dir)
    args.root_dir = root_dir
    data_type = args.data_type
    if data_type == "training":
        args.data_dir = args.train_data_dir
    elif data_type == "testing":
        args.data_dir = args.test_data_dir

    print("Current working dir: ", args.root_dir)
    features = args.features
    num_signals = args.num_signals
    per_of_samples = args.per_of_samples
    dataset = args.dataset_name
    batch_size = args.batch_size
    
    for time_step in args.time_steps:
        # Sep-up variable to define the AE model
        args.time_step = time_step
    
        feats_dict = {}
        df_windows_final = pd.DataFrame([])

        # # Repeating for each features....
        # Loading training dataset and extract features............

        #---------------------------------------------------------------#
        #                  STEP 1 Check and Generate Feats              #
        #---------------------------------------------------------------#
        load = False
        generate = True
        file_dir_dict = get_list_of_files(args)
        for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):

            #--------------------- creating a mapping of missing dataset -----------------------------
            windsizes_missing_dict = {} # Add the missing windsizes here......
            # Repeating for each features....
            df_windows_file = pd.DataFrame([])
            for feature_type in args.feature_type_list:
                print("feature_type: ", feature_type)
                windsizes_missing_dict[feature_type] = []
                # Loading training dataset and extract features............
                for windsize in args.windsizes: 
                    print("windsize: ", windsize)
                    
                    f_name = f"{args.feature_dir}/df_windows_tsfel_{dataset}_{file_name}_{per_of_samples}_{feature_type}_{windsize}.csv"
                    if file_exists(f_name):
                        print(f_name)
                        if load == True:
                            # Loading the file_wise extracted features........
                            df_windows_feature = pd.read_csv(f_name, index_col = 0) 
                            print("df_windows_file Loaded...")
                            #Adding to the final dataset......................
                            df_windows_file = pd.concat([df_windows_file, df_windows_feature], axis = 1, ignore_index = False)
                            feats_dict[feature_type] = df_windows_feature.columns
                            windsizes_missing_dict[feature_type].append(0)

                    else:
                        #Adding job list for later
                        if generate == True:
                            windsizes_missing_dict[feature_type].append(windsize)
                        else:
                            print("Skipped..")
                
            df_windows_final = pd.concat([df_windows_final, df_windows_file], axis = 0, ignore_index = True)
            #--------------------------------------------------------------------------------------


            if np.array(list(windsizes_missing_dict.values())).sum() == 0:
                print(f"All the features are extracted from {file_name}")
                
            else:
                print(f"Loading the features for windows", windsizes_missing_dict)
                print("Loading dataset: ",file_name)

                #--------------------- Remaining Files ----------------------
                try:
                    print("Starting loading ", file_index, file_name)
                    X_train, y_train = load_scale_data(args, file_name, file_path)
                except Exception as error:
                    print(error)
                    print(f"Skipping dataset {file_name}")

                # Repeating for each features....
                for feature_type in args.feature_type_list:

                    print("feature_type: ", feature_type)

                    # create cgf file....................................
                    cgf_file = tsfel.get_features_by_domain(feature_type) 
                
                    print(f"Extracted {feature_type} features: \n")
                    print(list(cgf_file[feature_type].keys()))

                    windsizes_missing = windsizes_missing_dict[feature_type].copy()

                    for windsize in windsizes_missing: 
                        print("windsize: ", windsize)

                        # Spliting in different batches........................
                        df_windows_file = pd.DataFrame([])
                        total_batch  = int(len(X_train)/batch_size)+1
                        for i in range (total_batch):
                            print("Extracting features - batch ", i+1, "//", total_batch)
                            #-----------------------------------------
                            y_cut = y_train[i*batch_size:(i+1)*batch_size].copy()
                            #------------------------------------------           
                            df_windows = tsfel.time_series_features_extractor(cgf_file, X_train[i*batch_size:(i+1)*batch_size], window_size=windsize, fs = 50, resample_rate= 100, n_jobs= -1)
                            df_windows['File'] = file_name
                            df_windows['Feature'] = feature_type
                            df_windows['Window'] = windsize
                            df_windows['Label'] = y_cut.rolling(windsize).max().dropna().iloc[::windsize].values

                            df_windows_file = pd.concat([df_windows_file, df_windows], ignore_index= True)

                        # Storing the file_wise extracted features........
                        df_windows_file.to_csv(f_name, index = True, header = True)

        
        #---------------------------------------------------------------#
        #                     STEP 2 - Merge Features                   #
        #---------------------------------------------------------------#
        load = True
        generate = False

        for feature_type in args.feature_type_list:
            print("feature_type: ", feature_type)

            df_windows_final = pd.DataFrame([])

            # # Repeating for each features....
            # Loading training dataset and extract features............
            for file_index, (file_name, file_path) in tqdm(enumerate(file_dir_dict.items())):

                df_windows_file = pd.DataFrame([])

                # Loading training dataset and extract features............
                for windsize in args.windsizes: 
                    print("windsize: ", windsize)
                    
                    f_name = f"{args.feature_dir}/df_windows_tsfel_{dataset}_{file_name}_{per_of_samples}_{feature_type}_{windsize}.csv"

                    if file_exists(f_name):
                        print(f_name)
                        if load == True:
                            # Loading the file_wise extracted features........
                            df_windows_feature = pd.read_csv(f_name, index_col = 0) 
                            print("df_windows_file Loaded...")
                            #Adding to the final dataset......................
                            df_windows_file = pd.concat([df_windows_file, df_windows_feature], axis = 1, ignore_index = False)
                    else:
                        #Adding job list for later
                        print("Skipped..")
                
                df_windows_final = pd.concat([df_windows_final, df_windows_file], axis = 0, ignore_index = True)
                print(feature_type, df_windows_final.shape)

            # Loadede.....
            # List of features.............
            feat_columns = sorted(list(set(df_windows_final.columns) - set(['File','Feature','Window','Label'])))
            df_windows_feats = df_windows_final[['File','Feature','Window', 'Label']].copy()

            print("Scaling the extracted features...........")
            scaler_filename = f"{args.scaler_dir}//scaler_data_{dataset}_{feature_type}_{windsize}.save"


            if data_type == 'training':
                scaler_data = MinMaxScaler()
                # scaler_data = StandardScaler()
                df_windows_final = scaler_data.fit_transform(df_windows_final[feat_columns].values)
                # Saving the scalers..........
                joblib.dump(scaler_data, scaler_filename) 
                
            else:
                scaler_data = joblib.load(scaler_filename) 
                df_windows_final = scaler_data.transform(df_windows_final[feat_columns].values)
                print("Test data scaled with train scalar")

            df_windows_final = pd.DataFrame(df_windows_final, columns = feat_columns)
            #--------------------------------------------------------------------------
            
        
            # Saving the dataset..........
            df_windows_final = pd.concat([df_windows_final, df_windows_feats], axis = 1)
            f_name = f"{args.feature_dir}/df_windows_tsfel_{dataset}_{file_name}_{per_of_samples}_{feature_type}_{windsize}.csv"

            df_windows_final.to_csv(f"{args.feature_dir}/df_windows_tsfel_clean_{dataset}_{data_type}_{feature_type}_{windsize}.csv", index = True, header = True)
            print(f"{args.feature_dir}/df_windows_tsfel_clean_{dataset}_{data_type}_{feature_type}_{windsize}.csv saved!!!")



            #---------------------------------------------------------------#
            #                 STEP 3 - Analyze variance                     #
            #---------------------------------------------------------------#
            # Get Variance of each features........
            df_windows_var = df_windows_final.var()
            ## Creating unique feaures from all the features -- remove signal id
            float_selected = list(set(df_windows_final.columns[df_windows_final.dtypes == float]))
            float_selected.remove("Label")
            list_of_unique_feat = list(set(["_".join(x.split("_")[1:]) for x in list(float_selected)]))

            # Creating dict of Features..... for panda dataframe...
            feat_dict = {}
            cgf_file = tsfel.get_features_by_domain() 

            for domain, domain_feat in cgf_file.items():
                for feature, feat_config in domain_feat.items():
                    count = 1
                    for unique_feat in list_of_unique_feat:
                        if feature in unique_feat:
                            feat_dict[unique_feat] = {}
                            feat_dict[unique_feat]['Domain'] = domain
                            feat_dict[unique_feat]['Feature'] = feature
                            feat_dict[unique_feat]['Feature no'] = count
                            count += 1
            # --------------
            feature_df = pd.DataFrame(feat_dict).T.sort_values(by = ['Domain', 'Feature'], ascending = True)
            feature_df    

            # ---
            # Adding data to the list of features......... Create variance map----
            for signal_no, signal_name in enumerate(features):
                list_of_features = [f"{signal_no}_{feat}" for feat in list_of_unique_feat]
                df_var = pd.DataFrame(df_windows_var[list_of_features].values, index = list_of_unique_feat, columns = [signal_name])
                feature_df = pd.concat([feature_df, df_var], axis = 1, ignore_index = False)   

            # ----------------------------------
            # Ploting the variance...
            #------------------------------------
            var_df = feature_df.groupby(['Feature']).max()[features]
            var_df = var_df.T[var_df.median(axis = 1).sort_values().index.to_list()]
            plt.figure(figsize = (12, 1.5))
            boxplot = var_df.boxplot()  
            plt.xticks(rotation = 90)
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_box_plot_{dataset}_{feature_type}.jpg")
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_box_plot_{dataset}_{feature_type}.pdf")
            plt.show()
            plt.show()
            #------------------------------------

            plt.figure(figsize = (12, 1.5))
            var_df.T.plot(linestyle = '--', marker = 'p', markersize = '3', figsize = (12, 1.5))
            plt.legend([])
            plt.xticks(rotation = 90)
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_line_plot_{dataset}_{feature_type}.jpg")
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_line_plot_{dataset}_{feature_type}.pdf")
            plt.show()
            #----------------------------------- 


            # List of features with different variance threshold...
            var_list = np.arange(0, 30, 2)/100
            count_df = pd.DataFrame([])
            for var in var_list:
                var_df_bin = (var_df >= var).astype(int)
                #----------------
                count_var = pd.DataFrame(var_df_bin.sum(), columns = ['Feature-wise Count'])
                count_var['Feature'] = count_var.index
                count_var['Variance Threshold'] = var
                #----------------
                count_df = pd.concat([count_df, count_var], axis = 0, ignore_index = True)
                #----------------

            count_df_total = count_df.groupby(['Variance Threshold']).sum()
            count_df_total['Variance Threshold'] = count_df_total.index


            count_df['Feature-wise Count'] = count_df['Feature-wise Count']/len(features)*100
            count_df_total['Feature-wise Count'] = count_df_total['Feature-wise Count'] #/1140*100

            # Plot --------------

            fig, ax = plt.subplots(1,1, figsize = (4, 4))
            sns.lineplot(count_df_total, x = 'Variance Threshold', y = 'Feature-wise Count', linestyle = '--', linewidth = '0.5', marker = 'p', markersize = '10', ax = ax)
            plt.grid(True)
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_vs_feature_count_dist_{dataset}_{feature_type}.jpg")
            plt.savefig(f"{args.plot_dir}/{dataset}/variance_vs_feature_count_dist_{dataset}_{feature_type}.pdf")
            plt.show()

            # plot ---------------

            fig, ax = plt.subplots(1,1, figsize = (6,5))
            sns.boxplot(count_df, x = 'Variance Threshold', y = 'Feature-wise Count', color = 'gray', ax = ax)
            sns.stripplot(count_df, x = 'Variance Threshold', y = 'Feature-wise Count', hue = 'Feature', ax = ax)
            plt.legend(bbox_to_anchor = (1,1), ncols = 3)
            plt.grid(True)
            plt.savefig(f"{args.plot_dir}/{dataset}/boxplot_variance_vs_feature_count_dist_{dataset}_{feature_type}.jpg")
            plt.savefig(f"{args.plot_dir}/{dataset}/boxplot_variance_vs_feature_count_dist_{dataset}_{feature_type}.pdf")
            plt.show()

            # --------------------
            sns.lineplot(count_df, x = 'Variance Threshold', y = 'Feature-wise Count', hue = 'Feature', linestyle = '--', linewidth = '0.5', marker = 'p', markersize = '10')
            plt.legend(bbox_to_anchor = (1,1.05), ncols = 3)
            plt.savefig(f"{args.plot_dir}/{dataset}/lineplot_variance_vs_feature_count_dist_{dataset}_{feature_type}.jpg")


            # -------------------
            # Saving the variance data...
            feature_df.to_csv(f"{args.features_dir}/feature_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv", header=True, index=True)
            var_df.to_csv(f"{args.features_dir}/variance_matrix_{dataset}_{data_type}_{feature_type}_{windsize}.csv", header=True, index=True)

if __name__ == "__main__":
    develop_canshield()




