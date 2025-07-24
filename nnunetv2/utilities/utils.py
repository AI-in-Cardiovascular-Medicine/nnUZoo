#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
from os.path import join, split
import re
from collections import defaultdict

from numba.cpython.listobj import list_to_list
from tqdm import tqdm
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from deep_utils import DirUtils


def get_identifiers_from_split_dataset_folder(folder: str, file_ending: str):
    print("Extracting file identifiers...")
    # files = subfiles(folder, suffix=file_ending, join=False)
    files = DirUtils.list_dir_full_path(folder, ends_with=file_ending)
    # all files have a 4 digit channel index (_XXXX)
    crop = len(file_ending) + 5
    files_cropped = defaultdict(list)
    for i in files:
        key = i[:-crop]
        if file_ending in key:
            print(f"[Warning] key: {key} still contains {file_ending}")
        files_cropped[key].append(i)
    # files = {}
    # only unique image ids

    # files_cropped = np.unique(files_cropped)
    return files_cropped.keys(), files_cropped


def create_lists_from_split_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None,
                                           files: List[str] = None) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    # if identifiers is None:
    #     identifiers = get_identifiers_from_split_dataset_folder(folder, file_ending)
    if files is None:
        # files = subfiles(folder, suffix=file_ending, join=False, sort=True)
        list_of_lists = [DirUtils.list_dir_full_path(folder, ends_with=file_ending)]
    if isinstance(files, dict):
        list_of_lists = list(files.values())
    # list_of_lists = []
    # for f in tqdm(identifiers, desc="create_list_from_split_dataset_folder"):
    #     p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
    #     list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists
    # return [files] #arthur, fix for nnUNet Imene8


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            dataset[k]['label'] = os.path.abspath(join(raw_dataset_folder, dataset[k]['label'])) if not os.path.isabs(
                dataset[k]['label']) else dataset[k]['label']
            dataset[k]['images'] = [os.path.abspath(join(raw_dataset_folder, i)) if not os.path.isabs(i) else i for i in
                                    dataset[k]['images']]
    else:
        identifiers, files = get_identifiers_from_split_dataset_folder(join(raw_dataset_folder, 'imagesTr'),
                                                                       dataset_json['file_ending'])
        images = create_lists_from_split_dataset_folder(join(raw_dataset_folder, 'imagesTr'),
                                                        dataset_json['file_ending'], identifiers, files)
        segs = [join(raw_dataset_folder, 'labelsTr', split(i)[-1] + dataset_json['file_ending']) for i in identifiers]
        dataset = dict()
        for i, im, se in zip(identifiers, images, segs):
            if os.path.exists(se):
                dataset[split(i)[-1]] = {'images': im, 'label': se}
            else:
                dataset[split(i)[-1]] = {'images': im, 'label': None}
    return dataset

if __name__ == '__main__':
    # output = get_filenames_of_train_images_and_targets("/media/aicvi/52cf80dd-fd23-4f1e-ac72-ab21abb33cc2/NNUNET_OUTPUT/nnunet_raw/Dataset037_pet_fm")
    # output_directory = "/media/aicvi/52cf80dd-fd23-4f1e-ac72-ab21abb33cc2/NNUNET_OUTPUT/nnunet_preprocessed/Dataset037_pet_fm/nnUNetPlans_3d_fullres"
    # from os.path import exists
    # for k in output:
    #     output_filename_truncated = join(output_directory, k)
    #     if (exists(output_filename_truncated + '.npz') or exists(output_filename_truncated + '.npy')) and exists(
    #         output_filename_truncated + '.pkl'):
    #         continue
    #
    #     if (exists(output_filename_truncated + '.npz') or exists(output_filename_truncated + '.npy')) and not exists(output_filename_truncated + '.pkl'):
    #         print(k, 'pkl missing...')
    #
    #     if not exists(output_filename_truncated + '.npy') and not exists(output_filename_truncated + '.pkl'):
    #         print(k, "npy and pkl missing...")
    output = get_filenames_of_train_images_and_targets(
        "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset201_ga_organs_03_03_2025_target")
    print(output)