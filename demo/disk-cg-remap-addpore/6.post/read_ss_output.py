import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("D:\\python_md_lib\\")

from lmp_output_io import *


files = ['disk_eta075.deform1ns', 'disk_eta080.deform1ns', 'disk_eta085.deform1ns','disk_eta090.deform1ns']
labels = ['75%', '80%', '85%','90%']
col_num = 10

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig1, ax1 = plt.subplots(figsize=(10,6))
all_strain = []
all_stress = []
for i, file_name in enumerate(files):
	print(i)
	indices, numbers = find_column_numbers_in_text(file_name, col_num)
	data = np.fromstring(''.join(numbers.split('\n')), sep=' ')
	data = data.reshape((len(indices), col_num))

	one_lenth = data[:, 2]
	one_strain = one_lenth/one_lenth[0]
	one_stress = data[:, 3]
	idx0, idx1 = deform_seg_indices(one_strain)
	strain_i = np.zeros(len(idx0))
	stress_i = np.zeros(len(idx0))
	for j in range(len(idx0)):
		strain_i[j] = np.average(one_strain[idx0[j]:idx1[j]])
		stress_i[j] = np.average(one_stress[idx0[j]:idx1[j]])
	all_strain.append(strain_i)
	all_stress.append(stress_i)
	ax1.plot(all_strain[-1][:-1]-1, -0.1*all_stress[-1][:-1], label=labels[i])
ax1.set_xlabel('Strain', fontsize=16)
ax1.set_ylabel('Stress (MPa)', fontsize=16)
ax1.legend()
plt.show()
