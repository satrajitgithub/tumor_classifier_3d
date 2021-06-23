import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like
from scipy import ndimage

from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, label_indices=None):
    """
    :param image_files:
    :param image_shape:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """

    label_indices = [label_indices]

    image_list = list()

    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        print("Reading: {}".format(image_file))
        image_list.append(read_image(image_file, image_shape=image_shape, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear'):
    if os.path.exists(os.path.abspath(in_file)):
        image = nib.load(os.path.abspath(in_file))

        image = fix_shape(image)  # Removes extra fourth axis if present
        return resize(image, new_shape=image_shape, interpolation=interpolation)

    else:
        print("[WARN] File does not exist. Zerofilling: ", os.path.abspath(in_file))
        return nib.Nifti1Image(np.zeros(image_shape), affine=None)

def check_unique_elements(np_array):
    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(np_array, return_counts=True)
    return str(dict(zip(unique, counts)))


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)
