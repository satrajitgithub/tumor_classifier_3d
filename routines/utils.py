import copy
import glob
import os
import numpy as np
import nibabel as nib

# **************************************************** Tumor ZOO **************************************************
tumor_zoo_params_dict_list = [{'class_tumor': 'HGG'},
                              {'class_tumor': 'LGG'},
                              {'class_tumor': 'METS'},
                              {'class_tumor': 'PITADE'},
                              {'class_tumor': 'ACSCHW'},
                              {'class_tumor': 'HEALTHY'},
                              {'class_tumor': 'MENINGIOMA'}]

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    new_header = header=nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, None, header=new_header)
    nib.save(output_nifti, output_path)

def create_training_validation_testing_files(logger, config, df, path_to_sessions, manual_label = None):
    training_files = list()
    subject_ids_tr = list()

    for subject_dir in path_to_sessions:
        subject_ids_tr.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + config["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_files.append(tuple(subject_files))

    training_files = [list(i) for i in training_files] # converting to list of lists from list of tuples

    logger.info("[SUBJECT_IDS] " + str(len(subject_ids_tr)) + " " + str(subject_ids_tr))

    if not manual_label:
        session_labels = np.array([df[df['sessions'] == i]['Tumor_type'].iloc[0] for i in subject_ids_tr])
    else:
        session_labels = [manual_label] * len(subject_ids_tr)

    assert len(session_labels) == len(subject_ids_tr)

    return training_files, session_labels, subject_ids_tr

def trim_df_by_dropping_nans(df, config):
    # Prepare the dataframe by reading from excel file

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df = df[df.scratch_path.notna()]  # There are some cases for which we do not have imaging data (i.e. NA in sessions column) - drop them

    return df

def trim_df_based_on_presence_in_scratch_and_modality(df, config):
    
    all_modalities = copy.deepcopy(config["all_modalities"])

    # List of all sessions of the worksheet : ['abspath/to/session1', 'abspath/to/session2', ..., 'abspath/to/sessionn']
    sessions_abspath_all = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in df.iterrows()]

    # True/False if session folder exists/doesnt in datapath : [True, False, ..., True]
    session_exists_logical = [os.path.isdir(i) for i in sessions_abspath_all]

    # Subset of sessions_abspath_all, containing only those sessions that exist
    sessions_abspath_exists_bef_modality_check = np.array(sessions_abspath_all)[np.array(session_exists_logical)].tolist()


    session_exists_modality_exists_logical_sublist = [[os.path.exists(os.path.join(i, j + ".nii.gz")) for j in all_modalities] for i in sessions_abspath_exists_bef_modality_check]

    # For each session, this gives True if all req modalities exist for that session
    session_exists_modality_exists_logical = [all(i) for i in session_exists_modality_exists_logical_sublist]

    # Use session_exists_modality_exists_logical indices to filter sessions_abspath_exists_bef_modality_check
    # This is the final list of sessions to be used
    sessions_abspath_exists = np.array(sessions_abspath_exists_bef_modality_check)[np.array(session_exists_modality_exists_logical)].tolist()

    df = df[df['sessions'].isin(os.path.basename(i) for i in (sessions_abspath_exists))]

    return df

def split_data_with_interal_testing_from_excel(config, info, df, sessions_abspath_exists):
    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.sessions.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions

    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    training_df = df[~df['fold'].str.contains(config['fold']) & ~df['fold'].str.contains('_test')]
    val_df = df[df['fold'].str.contains(config['fold'] + '_val')]
    test_df = df[df['fold'].str.contains('_test')]

    training_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in training_df.iterrows()]
    val_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in val_df.iterrows()]
    test_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in test_df.iterrows()]

    return training_session_abs_path, val_session_abs_path, test_session_abs_path




