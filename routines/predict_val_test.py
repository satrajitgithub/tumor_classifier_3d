import glob
import logging
import os
import random
import pandas as pd
from keras_contrib.layers import InstanceNormalization
import importlib
import matplotlib

from unet3d.data import write_data_to_file, open_data_file
from unet3d.utils import pickle_dump
from routines.utils import create_training_validation_testing_files

def run_validation_case_classification(df, logger, val_or_test, df_comps, truth_list, pred_list, data_index, output_dir,
                                       model, data_file, manual_label=None):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """

    test_data = np.asarray([data_file.root.data[data_index]])
    session_name = data_file.root.subject_ids[data_index].decode('utf-8')

    try:
        truth_clsfctn = df.loc[df['sessions'] == session_name][config['marker_column']].iloc[0]
    except:
        truth_clsfctn = manual_label
    # print("[DEBUG] truth_clsfctn", truth_clsfctn)

    class_label = config['labels_to_use'].index(truth_clsfctn)
    test_truth = keras.utils.to_categorical([class_label], num_classes=len(config['labels_to_use']))
    truth_list.append(test_truth)

    prediction = model.predict(test_data, verbose=0)
    pred_list.append(prediction)

    # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
    prediction_round = np.zeros_like(prediction)
    prediction_round[np.arange(len(prediction)), np.argmax(prediction)] = 1

    if np.array_equal(prediction_round, test_truth):
        verdict = "correct"
    else:
        verdict = "wrong"

    logger.info(
        ' {} \t --> truth = {}, prediction = {}, verdict = {}, confidence = {},'.format(os.path.basename(output_dir),
                                                                                        truth_clsfctn,
                                                                                        config["labels_to_use"][
                                                                                            np.argmax(prediction)],
                                                                                        verdict, np.amax(prediction)))

    confidence = np.max(prediction)
    rows, subject_ids = df_comps

    rows.append(
        [session_name, "fold" + config["fold"], val_or_test, truth_clsfctn, test_truth, np.around(prediction, 2),
         verdict, confidence])

    subject_ids.append(os.path.basename(output_dir))
    return rows, truth_list, pred_list


def run_validation_cases_classification(df, logger, config_file, model, hdf5_file, val_or_test="val",
                                        manual_label=None):
    config = config_file

    header = ("sessions", "fold", "val_or_test", "Type", "Truth", "Prediction", "Verdict", "Confidence")
    rows = list()
    subject_ids = list()
    truth_list_per_type = []
    pred_list_per_type = []

    # ###############################

    data_files = hdf5_file

    logger.info("\n" + "~" * 60 + "Predictions" + "~" * 60 + "\n")

    truth_list = []
    pred_list = []

    for data_file in data_files:
        data_file = tables.open_file(data_file, "r")

        count = 0

        for index, subj_id in enumerate(data_file.root.subject_ids):
            count += 1

            case_directory = subj_id.decode('utf-8')

            # print("Now doing:", case_directory)
            if config["seg_classify"] == 's_c' or config["seg_classify"] == 'c':
                rows, truth_list, pred_list = run_validation_case_classification(df, logger, val_or_test,
                                                                                 (rows, subject_ids), truth_list,
                                                                                 pred_list,
                                                                                 data_index=index,
                                                                                 output_dir=os.path.join(
                                                                                     config["basepath"], "predictions",
                                                                                     case_directory),
                                                                                 model=model,
                                                                                 data_file=data_file,
                                                                                 manual_label=manual_label)

        data_file.close()

    truth_list = np.squeeze(np.asarray(truth_list))
    pred_list = np.squeeze(np.asarray(pred_list))
    truth_list_per_type.append(truth_list)
    pred_list_per_type.append(pred_list)

    df2 = pd.DataFrame.from_records(rows, columns=header)
    df_merged = pd.merge(df, df2, on="sessions")
    df_merged.to_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_" + val_or_test + ".csv")

    truth_list_collapsed = [i for sublist in truth_list_per_type for i in sublist]
    pred_list_collapsed = [i for sublist in pred_list_per_type for i in sublist]
    np.save(config["basepath"] + "fold" + config["fold"] + "_" + val_or_test + '_truth.npy', truth_list_collapsed)
    np.save(config["basepath"] + "fold" + config["fold"] + "_" + val_or_test + '_pred.npy', pred_list_collapsed)

def get_pickle_lists(data_file, pickle_file_path):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file for training    
    :param pickle_file_path:
    """
    print("Creating pickle lists...")
    nb_samples_tr = data_file.root.data.shape[0] # Number of training data
    training_list = list(range(nb_samples_tr)) # List of integers: [0, 1, .. upto nb_samples_tr]            
    pickle_dump(training_list, pickle_file_path)

def load_old_model(model_file):
    custom_objects = {'InstanceNormalization': InstanceNormalization}
    return load_model(model_file, custom_objects=custom_objects)

def main(fold, exp):
    config_file_name="config_file_Exp"+exp
    config_file = importlib.import_module('config_files.'+config_file_name)
    set_fold = config_file.set_fold
    config = config_file.config
    info = config_file.info
    df = set_fold(fold, exp)

    # Create and configure logger
    log_path = os.path.join(config["basepath"], "prediction_log.txt")
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format=LOG_FORMAT,
                        level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    logging.getLogger('matplotlib.font_manager').disabled = True

    logger.info("*************************************************************************************************")
    logger.info("*" * 40 + " [ PREDICTION ] " + "*" * 40)
    logger.info("*************************************************************************************************")

        
    model_file = glob.glob(os.path.abspath(config["basepath"]+"modelClassifier_ep*.h5"))[0]
    logger.info("[INFO] Loading model from: {}".format(model_file))
    model = load_old_model(model_file)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info("\n" + "=" * 30 + " [TEST FILES] " + "=" * 30)
    test_files, test_labels, subject_ids_test, _ = create_training_validation_testing_files(logger,
                                                                                            config,
                                                                                            df,
                                                                                            path_to_sessions=config[
                                                                                                "testing_sessions"])

    for i, tumor in enumerate(config["tumor_type"]):

        logger.info("\n" + "~" * 60 + "  " + tumor + "  " + "~" * 60)

        print("\n", "=" * 30, tumor, ": [TESTING] write_data_to_file", "=" * 30, "\n")
        write_data_to_file(config, np.array(test_files)[test_labels == tumor].tolist(),
                           config[i]["data_file_test"],
                           subject_ids=np.array(subject_ids_test)[test_labels == tumor].tolist())

    hdf5_file_list = [config[i]["data_file_test"] for i in range(len(config["tumor_type"]))]

    logger.info("hdf5_file_list: " + str(hdf5_file_list))

    run_validation_cases_classification(df, logger, config, model, hdf5_file_list, "test")

