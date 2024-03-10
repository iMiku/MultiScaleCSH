#!/usr/bin/env python
# coding: utf-8

# In[1]:

from ovito.io import import_file, export_file
from ovito.modifiers import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import functools
import pyny3d.geoms as pyny
import networkx as nx

def csh_composition_count(file_name):
    # analyze chemical formula of csh
    # 1 40.077999  # Ca1 skeleton calcium
    # 2 40.077999  # Ca2 free calcium
    # 3 1.008      # H3 water hydrogen
    # 4 1.008      # H4 hydroxide H
    # 5 15.999     # O5 water oxygen
    # 6 15.999     # O6 bridge O between silicon atoms
    # 7 15.999     # O7 hydroxide oxygen
    # 8 28.085501  # Si8 silicon 
    # 9 12.0107    # C9 external carbon
    pipeline = import_file(file_name)
    data = pipeline.compute()
    mass_type1_to_type9 = [40.077999, 40.077999, 1.008, 1.008, 15.999, 15.999, 15.999, 28.085501, 12.0107]
    num_type1_to_type9 = [len(np.where(data.particles['Particle Type']==(i+1))[0]) for i in range(9)]
    num_Ca_total = num_type1_to_type9[0] + num_type1_to_type9[1]
    num_Si = num_type1_to_type9[7]
    num_wat = num_type1_to_type9[4]
    num_OH = num_type1_to_type9[3]
    volume = data.cell[0,0]*data.cell[1,1]*data.cell[2,2]
    total_mass = np.sum([num_type1_to_type9[i]*mass_type1_to_type9[i] for i in range(len(mass_type1_to_type9))])
    density = total_mass/volume*1.0/6.02e23*1e24 # g/cm^3
    print('Density: ',density,' g/cm3')
    print('Ca/Si: ', num_Ca_total/num_Si)
    print('Water mass ratio: ', num_wat*18/total_mass)

def csh_3_kind_water_count(file_name):
    # assuming the model is made of csh (first half) and additional water (second half)
    pipeline = import_file(file_name)

    data = pipeline.compute()
    mol_id_max = np.max(data.particles['Molecule Identifier'])
    mol_id_mid = mol_id_max/2
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'MoleculeIdentifier<=%d'%int(mol_id_mid)))
    pipeline.modifiers.append(ComputePropertyModifier(
                              output_property = 'is_csh',
                              expressions = '1',
                              only_selected = True                 )
                             )
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'MoleculeIdentifier> %d'%int(mol_id_mid)))
    pipeline.modifiers.append(ComputePropertyModifier(
                              output_property = 'is_inner_wat',
                              expressions = '0',
                              only_selected = True,
                              neighbor_expressions = 'is_csh',
                              cutoff_radius = 3.0                 )
                              )
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'is_inner_wat> 0'))
    pipeline.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Bonded,iterations=3))
    pipeline.modifiers.append(ComputePropertyModifier(
                              output_property = 'is_inner_wat',
                              expressions = '1',
                              only_selected = True                 )
                             )
    data = pipeline.compute()
    id_csh_all = np.where(data.particles['Molecule Identifier']<=mol_id_mid)[0]
    id_wat_all = np.union1d(np.where(data.particles['Particle Type']==3)[0], np.where(data.particles['Particle Type']==5)[0])
    id_wat_embeded = np.intersect1d(id_wat_all, id_csh_all)
    id_wat_added = np.where(data.particles['Molecule Identifier'] >mol_id_mid)[0]
    id_wat_inner = np.where(data.particles['is_inner_wat']>0)[0]
    id_wat_outer = np.setdiff1d(id_wat_added, id_wat_inner)

    mass_type1_to_type9 = [40.077999, 40.077999, 1.008, 1.008, 15.999, 15.999, 15.999, 28.085501, 12.0107]
    mass_csh_all = np.sum(data.particles['Mass'][id_csh_all])
    mass_wat_inner = np.sum(data.particles['Mass'][id_wat_inner])
    mass_wat_outer = np.sum(data.particles['Mass'][id_wat_outer])
    mass_wat_embeded = np.sum(data.particles['Mass'][id_wat_embeded])
    mass_total = np.sum(data.particles['Mass'])
    print('Mass ratio embeded water/csh: ',mass_wat_embeded/mass_csh_all)
    print('Mass ratio   inner water/csh: ',mass_wat_inner  /mass_csh_all)
    print('Mass ratio   outer water/csh: ',mass_wat_outer  /mass_csh_all)
    pass
