import os
import random
import copy
import numpy as np
import pandas as pd

from routines.utils import *
from unet3d.model_classification import *

config = dict()
info = dict()

# Set the molecular parameter for this experiment
config["tumor_type"] = [i['class_tumor'] for i in tumor_zoo_params_dict_list]

for i, tumor_dict in enumerate(tumor_zoo_params_dict_list):
    config[i] = tumor_dict

config['labels_to_use'] = config["tumor_type"]
config['marker_column'] = 'Tumor_type'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config['path_to_data'] = '/scratch/satrajit/data/'

config['excel_path'] = "/home/satrajit/tumor_classification_revamped/tumor_classification.xlsx"

config["all_modalities"] = ["T1c_subtrMeanDivStd"]

config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["truth"] = ["OTMultiClass"]
config["nb_channels"] = len(config["training_modalities"])
config["truth_channel"] = config["nb_channels"]

config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))  # (1,128,128,128)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["model"] = (isensee2017_classification,)
config["n_base_filters"] = 16
config["network_depth"] = 5
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["initial_learning_rate"] = 0.0005  # 0.001, 0.0005, 0.00025
config["network_depth"] = 5
config['loss_function'] ='categorical_crossentropy'

config['regularizer'] = regularizers.l2(1e-5)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["batch_size"] = 5
config["validation_batch_size"] = 5
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Augmentation parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For volume data: specify one or more of [0,1,2] eg: [0], [0,1], [1,2], [0,1,2] etc
config["flip"] = [0, 1, 2]  # augments the data by randomly flipping an axis during training
config["augment"] = config["flip"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File paths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.

# Split data into 5 folds from excel
def set_fold(fold, exp):
    config["fold"] = fold

    # Setting the basepath of the folder inside which everything will be stored: molecular/results/<experiment>/<fold>/
    config["basepath"] = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "fold" + fold + "/"

    # Read excel
    df = pd.read_excel(config['excel_path'])

    # Trim excel by removing entries containing Nans
    df = trim_df_by_dropping_nans(df, config)
    df = trim_df_based_on_presence_in_scratch_and_modality(df, config)
    df_after_dropping_nans = copy.deepcopy(df)

    # Trim cases based on availability of all modalities including GT
    sessions_abspath_exists = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in df.iterrows()]

    # Split data into n folds from excel
    config['training_sessions'], config['validation_sessions'], config['testing_sessions'] = split_data_with_interal_testing_from_excel(config, info, df, sessions_abspath_exists)

    info['training_sessions'] = [os.path.basename(sess) for sess in config['training_sessions']]
    y_train = df.loc[df.sessions.isin(info['training_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_train_unique, y_train_count = np.unique(y_train, return_counts=True)
    info['class_distribution_training'] = str(dict(zip(y_train_unique, y_train_count)))

    info['validation_sessions'] = [os.path.basename(sess) for sess in config['validation_sessions']]
    y_val = df.loc[df.sessions.isin(info['validation_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_val_unique, y_val_count = np.unique(y_val, return_counts=True)
    info['class_distribution_validation'] = str(dict(zip(y_val_unique, y_val_count)))

    info['testing_sessions'] = [os.path.basename(sess) for sess in config['testing_sessions']]
    y_test = df.loc[df.sessions.isin(info['testing_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_test_unique, y_test_count = np.unique(y_test, return_counts=True)
    info['class_distribution_testing'] = str(dict(zip(y_test_unique, y_test_count)))

    # *************************************************** Set dataset for external testing *************************************************

    # Any additional conditions to filter data will come here
    df_ext = df_after_dropping_nans[(df_after_dropping_nans.fold.str.contains("new"))]
    # End

    info['ext_sessions_before_modality_omission'] = df_ext['sessions'].tolist()

    # [Conditional] Trim cases based on requirement of OTMultiClass and T1c
    df_ext = trim_df_based_on_GT(df_ext, config)
    df_ext = trim_df_based_on_Tumor_modality(df_ext, config)

    # Trim cases based on availability in scratch directory
    sessions_abspath_exists_bef_modality_check = filter_sessions_based_on_availability_in_scratch(df_ext, config)

    # Trim cases based on availability of all modalities including GT
    sessions_abspath_exists = filter_sessions_based_on_availability_of_modalities(sessions_abspath_exists_bef_modality_check, config)

    info['ext_sessions_after_modality_omission'] = [os.path.basename(i) for i in sessions_abspath_exists]
    info['ext_sessions_omitted_in_modality_omission'] = \
        list(set(info['ext_sessions_before_modality_omission']).difference(set(info['ext_sessions_after_modality_omission'])))


    config['ext_sessions'] = sessions_abspath_exists

    info['ext_sessions'] = [os.path.basename(sess) for sess in config['ext_sessions']]
    y_test = df_after_dropping_nans.loc[df_after_dropping_nans.sessions.isin(info['ext_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_test_unique, y_test_count = np.unique(y_test, return_counts=True)
    info['class_distribution_ext'] = str(dict(zip(y_test_unique, y_test_count)))

    
    # ***************************************************************************************************************************
    for i, tumor in enumerate(config["tumor_type"]):
        config[i]["data_file_tr"] = os.path.abspath(config["basepath"]+tumor+"_data_tr.h5")
        config[i]["data_file_val"] = os.path.abspath(config["basepath"]+tumor+"_data_val.h5")
        config[i]["training_file"] = os.path.abspath(config["basepath"]+tumor+"_training_ids.pkl")

        config[i]["validation_file"] = os.path.abspath(config["basepath"]+tumor+"_validation_ids.pkl")
        config[i]["data_file_test"] = os.path.abspath(config["basepath"]+tumor+"_data_test.h5")
        config[i]["testing_file"] = os.path.abspath(config["basepath"]+tumor+"_testing_ids.pkl")
        
    # Path to which external test hdf5 files will be written to
    config["data_file_ext"] = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "data_ext.h5"
    config["ext_file"] = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "ext_ids.pkl"

    return df_after_dropping_nans



