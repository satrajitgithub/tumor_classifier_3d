import copy
import os

import numpy as np
import tables
from unet3d.utils import resize, read_image_files

def reslice_image_set(in_files, image_shape, label_indices=None):
    images = read_image_files(in_files, image_shape=image_shape, label_indices=label_indices)

    return images

def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    # https://github.com/ellisdg/3DUnetCNN/issues/58#issuecomment-415884226
    # filters = tables.Filters(complevel=0) # uncomment this if above not working
    data_shape = tuple([0, n_channels] + list(image_shape)) # (#Samples = 0,#C,H,W,D)
    truth_shape = tuple([0, 1] + list(image_shape)) # (#Samples = 0,1,H,W,D)
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape, filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4), filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage, truth_dtype=np.uint8):
    '''
    :param image_files: list of lists where each sublist corresponds to one subject
    '''
    for idx, set_of_files in enumerate(image_files):
        # set_of_files is the set of modalities + GT for each subject

        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1)

        subject_data = [image.get_fdata() for image in images]

        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels, truth_dtype)

    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):

    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def write_data_to_file(training_data_files,
                       out_file,
                       image_shape,
                       truth_dtype=np.uint8,
                       subject_ids=None):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of lists containing the training data files. The modalities should be listed in
    the same order in each sublist. The last item in each sublist must be the labeled image. 
    Example: [['sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'], 
              ['sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz'],
							              :
							              :
              ['subn-T1.nii.gz', 'subn-T2.nii.gz', 'subn-truth.nii.gz']]

    :param out_file: name of the hdf5 file where data will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1

    hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)


    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape, truth_dtype=truth_dtype,
                            affine_storage=affine_storage)
    hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)

    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
