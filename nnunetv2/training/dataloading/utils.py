from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def find_broken_image_and_labels(
    path_to_data_dir: str | Path,
) -> tuple[set[str], set[str]]:
    """
    Iterates through all numpys and tries to read them once to see if a ValueError is raised.
    If so, the case id is added to the respective set and returned for potential fixing.

    :path_to_data_dir: Path/str to the preprocessed directory containing the npys and npzs.
    :returns: Tuple of a set containing the case ids of the broken npy images and a set of the case ids of broken npy segmentations. 
    """
    content = os.listdir(path_to_data_dir)
    unique_ids = [c[:-4] for c in content if c.endswith(".npz")]
    failed_data_ids = set()
    failed_seg_ids = set()
    for unique_id in unique_ids:
        # Try reading data
        try:
            np.load(path_to_data_dir / (unique_id + ".npy"), "r")
        except ValueError:
            failed_data_ids.add(unique_id)
        # Try reading seg
        try:
            np.load(path_to_data_dir / (unique_id + "_seg.npy"), "r")
        except ValueError:
            failed_seg_ids.add(unique_id)

    return failed_data_ids, failed_seg_ids


def try_fix_broken_npy(path_do_data_dir: Path, case_ids: set[str], fix_image: bool):
    """ 
    Receives broken case ids and tries to fix them by re-extracting the npz file (up to 5 times).

    :param case_ids: Set of case ids that are broken.
    :param path_do_data_dir: Path to the preprocessed directory containing the npys and npzs.
    :raises ValueError: If the npy file could not be unpacked after 5 tries. --
    """
    for case_id in case_ids:
        for i in range(5):
            try:
                key = "data" if fix_image else "seg"
                suffix = ".npy" if fix_image else "_seg.npy"
                read_npz = np.load(path_do_data_dir / (case_id + ".npz"), "r")[key]
                np.save(path_do_data_dir / (case_id + suffix), read_npz)
                # Try loading the just saved image.
                np.load(path_do_data_dir / (case_id + suffix), "r")
                break
            except ValueError:
                if i == 4:
                    raise ValueError(
                        f"Could not unpack {case_id + suffix} after 5 tries!"
                    )
                continue


def verify_or_stratify_npys(path_to_data_dir: str | Path) -> None:
    """
    This re-reads the npy files after unpacking. Should there be a loading issue with any, it will try to unpack this file again and overwrites the existing.
    If the new file does not get saved correctly 5 times, it will raise an error with the file name to the user. Does the same for images and segmentations.
    :param path_to_data_dir: Path to the preprocessed directory containing the npys and npzs.
    :raises ValueError: If the npy file could not be unpacked after 5 tries. --
      Otherwise an obscured error will be raised later during training (depending when the broken file is sampled)
    """
    path_to_data_dir = Path(path_to_data_dir)
    # Check for broken image and segmentation npys
    failed_data_ids, failed_seg_ids = find_broken_image_and_labels(path_to_data_dir)

    if len(failed_data_ids) != 0 or len(failed_seg_ids) != 0:
        warn(
            f"Found {len(failed_data_ids)} faulty data npys and {len(failed_seg_ids)}!\n"
            + f"Faulty images: {failed_data_ids}; Faulty segmentations: {failed_seg_ids})\n"
            + "Trying to fix them now."
        )
        # Try to fix the broken npys by reextracting the npz. If that fails, raise error
        try_fix_broken_npy(path_to_data_dir, failed_data_ids, fix_image=True)
        try_fix_broken_npy(path_to_data_dir, failed_seg_ids, fix_image=False)


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt



def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        if len(npz_files):
            p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify] * len(npz_files))
                  )
        else:
            print("[INFO] There are no .npz files. Maybe SSL and already data is saved!")

def get_case_identifiers(folder: str, extension: str = "npz") -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(extension) and (i.find("segFromPrevStage") == -1)]
    return case_identifiers
