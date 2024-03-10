#!/usr/bin/env python
# coding: utf-8
# import importlib.util

# # Absolute path of the Python file you want to import
# absolute_path = 'D:\\python_md_lib\\cg2aa_funcs.py'
# # Create a module specification from the absolute path
# module_spec = importlib.util.spec_from_file_location('cg2aa_funcs', absolute_path)
# # Load the module using the module specification
# cg2aa = importlib.util.module_from_spec(module_spec)
# module_spec.loader.exec_module(cg2aa)

import sys
sys.path.append("D:\\python_md_lib\\")

from cg2aa_funcs import *

#file1, box = convert_from_nano_to_scaleX('202002_3x3x3.340000.lammpstrj', scale_factor=6.37)
#file1, box = convert_from_nano_to_scaleX('202002_4x4x4.500000.lammpstrj', scale_factor=5.20)
file1, box = convert_from_nano_to_scaleX('202002_5x5x5.500000.lammpstrj', scale_factor=4.17)
#file1, box = convert_from_nano_to_scaleX('202002_6x6x6.500000.lammpstrj', scale_factor=3.46)
#file1, box = convert_from_nano_to_scaleX('202002_7x7x7.500000.lammpstrj', scale_factor=2.93)

file2 = compute_quat_to_largest_face_in_voro_cell(file1)
create_parts_aa_ref_cg_voroquat('382530_AA.data',file2, vol_cut=-1, wat_file='wat1000_AA.data', 
    slice_dist_scale=0.99999, slice_gap_min=1.5)
file3 = merge_regions([125, 25, 1])

refine_box_ff_section(file3, box)
reset_molecule_id_from_1_to_n(file3, file3)
print("obtained data file: "+file3)

print("cleanning")
import os
import glob
directory_path = '.'  # Replace with the path to your directory
files_to_keep = [file3, '382550_AA.data', '382530_AA.data', 'wat1000_AA.data']  # List of files to keep

# Get a list of all files in the directory
all_files = glob.glob(os.path.join(directory_path, '*.data'))

# Iterate over the files in the directory
for file_path in all_files:
    file_name = os.path.basename(file_path)
    if file_name not in files_to_keep:
        # Delete files that are not in the list
        os.remove(file_path)
