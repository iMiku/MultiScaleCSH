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

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def quaternion_from_two_vectors(v_from, v_to):
    """
    Calculate the quaternion for rotation from vector v_from to vector v_to.
    Parameters:
        v_from (numpy array): 3D vector (x, y, z) representing the starting vector.
        v_to (numpy array): 3D vector (x, y, z) representing the target vector.

    Returns:
        numpy array: Quaternion (x, y, z, w) representing the rotation.
    """
    v_from = np.asarray(v_from)
    v_to = np.asarray(v_to)
    quaternion = R.from_matrix(rotation_matrix_from_vectors(v_from, v_to)).as_quat()
    return quaternion

def ovito_modify_assign_to_data(frame, data, property_name='new_property', property_values=[]):
    data.particles_.create_property(property_name, data=property_values)

def compute_quat_to_largest_face_in_voro_cell(file_name):
    pipeline = import_file(file_name)
    pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices = True, generate_polyhedra=True))
    data = pipeline.compute()
    box_size = [data.cell[0][0], data.cell[1][1], data.cell[2][2]]
    surface = data.surfaces['voronoi-polyhedra']
    vertex_coords = surface.vertices['Position'] 
    particle_ids = surface.regions['Particle Identifier']
    new_quaternions = []
    for region_id in range(len(particle_ids)):
        p_index = np.where(data.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        center_pos = data.particles['Position'][p_index]
        all_face_ids = get_face_id_list_in_region(region_id, surface)
        largest_area = 0
        largest_area_face_id = 0
        for face_id in all_face_ids:
            vertices = vertex_coords[get_all_vertices(face_id, surface)]
            vertices_close = np.array([closest_point2ref_in_periodic_box(v, center_pos, box_size) for v in vertices])
            pyny.Polygon.verify = False
            face_area = pyny.Polygon(vertices_close).get_area()
            if(face_area > largest_area):
                largest_area_face_id = face_id
                largest_area = face_area
        print(region_id, largest_area_face_id, largest_area)
        p3 = vertex_coords[get_3vertex_id_list_of_face(largest_area_face_id, surface)]
        p3[0,:] = closest_point2ref_in_periodic_box(p3[0,:], center_pos, box_size)
        p3[1,:] = closest_point2ref_in_periodic_box(p3[1,:], center_pos, box_size)
        p3[2,:] = closest_point2ref_in_periodic_box(p3[2,:], center_pos, box_size)
        p1 = center_pos
        vec, dist = calc_normal_vec(p3, p1)
        quat_new = quaternion_from_two_vectors([0,0,1], vec)
        new_quaternions.append(quat_new)
    new_quaternions = np.asarray(new_quaternions)
    custom_modify0 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_x', property_values=new_quaternions[:,0])
    custom_modify1 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_y', property_values=new_quaternions[:,1])
    custom_modify2 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_z', property_values=new_quaternions[:,2])
    custom_modify3 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_w', property_values=new_quaternions[:,3])
    pipeline.modifiers.append(custom_modify0)
    pipeline.modifiers.append(custom_modify1)
    pipeline.modifiers.append(custom_modify2)
    pipeline.modifiers.append(custom_modify3)
    out_name = "add_voro_quat_"+file_name
    export_file(pipeline, out_name, "lammps/dump", columns =
  ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", 
   "c_q1", "c_q2", "c_q3", "c_q4", "voro_quat_x", "voro_quat_y", "voro_quat_z", "voro_quat_w"])
    return out_name

def compute_quat_to_largest_face_in_voro_cell_gcmc(file_name):
    pipeline = import_file(file_name)
    pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices = True, generate_polyhedra=True))
    data = pipeline.compute()
    box_size = [data.cell[0][0], data.cell[1][1], data.cell[2][2]]
    surface = data.surfaces['voronoi-polyhedra']
    vertex_coords = surface.vertices['Position'] 
    particle_ids = surface.regions['Particle Identifier']
    new_quaternions = []
    for region_id in range(len(particle_ids)):
        p_index = np.where(data.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        center_pos = data.particles['Position'][p_index]
        all_face_ids = get_face_id_list_in_region(region_id, surface)
        largest_area = 0
        largest_area_face_id = 0
        for face_id in all_face_ids:
            vertices = vertex_coords[get_all_vertices(face_id, surface)]
            vertices_close = np.array([closest_point2ref_in_periodic_box(v, center_pos, box_size) for v in vertices])
            pyny.Polygon.verify = False
            face_area = pyny.Polygon(vertices_close).get_area()
            if(face_area > largest_area):
                largest_area_face_id = face_id
                largest_area = face_area
        print(region_id, largest_area_face_id, largest_area)
        p3 = vertex_coords[get_3vertex_id_list_of_face(largest_area_face_id, surface)]
        p3[0,:] = closest_point2ref_in_periodic_box(p3[0,:], center_pos, box_size)
        p3[1,:] = closest_point2ref_in_periodic_box(p3[1,:], center_pos, box_size)
        p3[2,:] = closest_point2ref_in_periodic_box(p3[2,:], center_pos, box_size)
        p1 = center_pos
        vec, dist = calc_normal_vec(p3, p1)
        quat_new = quaternion_from_two_vectors([0,0,1], vec)
        new_quaternions.append(quat_new)
    new_quaternions = np.asarray(new_quaternions)
    custom_modify0 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_x', property_values=new_quaternions[:,0])
    custom_modify1 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_y', property_values=new_quaternions[:,1])
    custom_modify2 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_z', property_values=new_quaternions[:,2])
    custom_modify3 = functools.partial(ovito_modify_assign_to_data, property_name='voro_quat_w', property_values=new_quaternions[:,3])
    pipeline.modifiers.append(custom_modify0)
    pipeline.modifiers.append(custom_modify1)
    pipeline.modifiers.append(custom_modify2)
    pipeline.modifiers.append(custom_modify3)
    out_name = "add_voro_quat_"+file_name
    export_file(pipeline, out_name, "lammps/dump", columns =
  ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", 
   "voro_quat_x", "voro_quat_y", "voro_quat_z", "voro_quat_w"])
    return out_name

def convert_from_nano_to_scaleX(file_name, scale_factor=10.0):
    pipeline = import_file(file_name)
    scale = AffineTransformationModifier(
          #operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[scale_factor,  0,  0, 0],
                            [ 0, scale_factor,  0, 0],
                            [ 0,  0, scale_factor, 0]],
          only_selected = False)
    pipeline.modifiers.append(scale)  
    data = pipeline.compute()
    box_lohi = [ [ data.cell[0,3], data.cell[0,3] + data.cell[0,0] ], 
                 [ data.cell[1,3], data.cell[1,3] + data.cell[1,1] ], 
                 [ data.cell[2,3], data.cell[2,3] + data.cell[2,2] ] ] 
    out_name = "scale%.2f_"%(scale_factor)+file_name
    print(out_name)
    export_file(pipeline, out_name, "lammps/dump", columns =
  ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", 
   "c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])
    return out_name, box_lohi

def convert_from_nano_to_scaleX_gcmc(file_name, scale_factor=10.0):
    pipeline = import_file(file_name)
    scale = AffineTransformationModifier(
          #operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[scale_factor,  0,  0, 0],
                            [ 0, scale_factor,  0, 0],
                            [ 0,  0, scale_factor, 0]],
          only_selected = False)
    pipeline.modifiers.append(scale)  
    data = pipeline.compute()
    box_lohi = [ [ data.cell[0,3], data.cell[0,3] + data.cell[0,0] ], 
                 [ data.cell[1,3], data.cell[1,3] + data.cell[1,1] ], 
                 [ data.cell[2,3], data.cell[2,3] + data.cell[2,2] ] ] 
    out_name = "scale%.2f_"%(scale_factor)+file_name
    print(out_name)
    export_file(pipeline, out_name, "lammps/dump", columns =
  ["Particle Identifier", "Particle Type", "Charge", "Position.X", "Position.Y", "Position.Z"])
    return out_name, box_lohi

def convert_from_nano_to_real(file_name):
    pipeline = import_file(file_name)
    scale = AffineTransformationModifier(
          #operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[10,  0,  0, 0],
                            [ 0, 10,  0, 0],
                            [ 0,  0, 10, 0]],
          only_selected = False)
    pipeline.modifiers.append(scale)  
    data = pipeline.compute()
    box_lohi = [ [ data.cell[0,3], data.cell[0,3] + data.cell[0,0] ], 
                 [ data.cell[1,3], data.cell[1,3] + data.cell[1,1] ], 
                 [ data.cell[2,3], data.cell[2,3] + data.cell[2,2] ] ] 
    out_name = "unitsA_"+file_name
    print(out_name)
    export_file(pipeline, out_name, "lammps/dump", columns =
  ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", 
   "c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])
    return out_name, box_lohi

def get_face_id_list_in_region(region_id, surface):
    faces_region = np.array(surface.faces['Region'])
    face_id_list = np.where(faces_region==region_id)[0]
    return face_id_list

def get_all_edges(face_id, surface):
    start_edge = surface.topology.first_face_edge(face_id)
    count = 1
    edge = surface.topology.next_face_edge(start_edge)
    all_edges = [edge]
    while edge != start_edge:
        assert surface.topology.adjacent_face(edge) == face_id
        count += 1
        edge = surface.topology.next_face_edge(edge)
        all_edges.append(edge)
    return all_edges

def get_all_vertices(face_id, surface):
    all_edges = get_all_edges(face_id, surface)
    all_vertices = surface.topology.first_edge_vertex(all_edges)
    return all_vertices

def get_3vertex_id_list_of_face(face_id, surface):
    topo = surface.topology
    edge1_id = topo.first_face_edge(face_id)
    edge2_id = topo.next_face_edge(edge1_id)
    vertex1_id = topo.first_edge_vertex(edge1_id)
    vertex2_id = topo.first_edge_vertex(edge2_id)
    vertex3_id = topo.second_edge_vertex(edge2_id)
    return np.array([vertex1_id, vertex2_id, vertex3_id])

def calc_normal_vec(p3_face, p1_inside):
    # Calculate two vectors from the given points
    vector1 = np.subtract(p3_face[0], p3_face[1])
    vector2 = np.subtract(p3_face[1], p3_face[2])
    # Calculate the cross product of the two vectors
    normal_vector = np.cross(vector1, vector2)
    # Normalize the resulting vector to get the unit normal vector
    normal_vector /= np.linalg.norm(normal_vector)
    # Dot product between the normal vector and a vector from one of the face vertices to the test point
    dot_product = np.dot(normal_vector, np.subtract(p3_face[0], p1_inside))
    # If the dot product is negative, it means the normal vector points towards the test point
    if(dot_product < 0):
        normal_vector *= -1
        dot_product *= -1
    distance = dot_product
    return normal_vector, distance

def closest_point2ref_in_periodic_box(point, ref, box):
    periodic_image = [-1, 0, 1]
    closest_dist  = np.linalg.norm(point - ref)
    closest_point = point
    for ix in periodic_image:
        for iy in periodic_image:
            for iz in periodic_image:
                new_point = point + np.array([ix, iy, iz])*np.array(box)
                dist = np.linalg.norm(new_point - ref)
                if(dist < closest_dist):
                    closest_dist = dist
                    closest_point = new_point
    return closest_point

def rep_rot_slice_move(cell_file, cg_box_size, quat, surface_mesh, region_id, slice_dist_scale, new_center, out_name, slice_gap_min=1.5):
    # cell_file: str, name of the all-atom model used as an unit cell
    # cg_box_size: [1 x 3] box size of the cg model
    # quat: [w x y z] quaternion for rotation
    # surface_mesh: ovito.data.SurfaceMesh
    # region_id: int, id of the region
    # slice_dist_scale: float, a scaled offset to do slice
    # new_center: [x y z] center of mass
    # out_name: str, name of the output file
    pipeline = import_file(cell_file)
    ### 1. apply replication
    rep_t = [9, 9, 9]
    pipeline.modifiers.append(ReplicateModifier(num_x=rep_t[0], num_y=rep_t[1],num_z=rep_t[2]))
    ### 2. apply rotation 
    a1 = R.from_quat(quat).as_matrix()
    rotate = AffineTransformationModifier(
          operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[a1[0,0], a1[0,1], a1[0,2], 0],
                            [a1[1,0], a1[1,1], a1[1,2], 0],
                            [a1[2,0], a1[2,1], a1[2,2], 0]],
          only_selected = False)
    pipeline.modifiers.append(rotate)
    ### 3. apply slices
    ##### 3.1. get id of all faces
    face_ids = get_face_id_list_in_region(region_id, surface_mesh)
    ##### 3.2. enumerate face id and apply slice mod
    vertex_coords = surface_mesh.vertices['Position']
    for face_id in face_ids:
        p3 = vertex_coords[get_3vertex_id_list_of_face(face_id, surface_mesh)]
        p3[0,:] = closest_point2ref_in_periodic_box(p3[0,:], new_center, cg_box_size)
        p3[1,:] = closest_point2ref_in_periodic_box(p3[1,:], new_center, cg_box_size)
        p3[2,:] = closest_point2ref_in_periodic_box(p3[2,:], new_center, cg_box_size)
        p1 = new_center
        vec, dist = calc_normal_vec(p3, p1)
        relaxed_dist = dist*slice_dist_scale
        if( relaxed_dist > (dist - slice_gap_min) ):
            relaxed_dist = dist - slice_gap_min
            if(relaxed_dist < 0):
                relaxed_dist = 0
        one_slice = SliceModifier(normal=vec, distance=relaxed_dist, select=True)
        pipeline.modifiers.append(one_slice)
        pipeline.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Bonded,iterations=3))
        pipeline.modifiers.append(DeleteSelectedModifier())
    ### 4. apply move
    move = AffineTransformationModifier(
          operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[1, 0, 0, new_center[0]],
                            [0, 1, 0, new_center[1]],
                            [0, 0, 1, new_center[2]]],
          only_selected = False)
    pipeline.modifiers.append(move)
    ### 5. change molecule to region_id
    pipeline.modifiers.append(ComputePropertyModifier(
        output_property = 'Molecule Identifier',
        expressions = '%d'%(region_id+1)               )
                             )
    # Export combined dataset to a new file.
    export_file(pipeline, out_name, "lammps/data", atom_style="full")

def rep_rot_slice_shell_move(cell_file, cg_box_size, quat, surface_mesh, region_id, slice_dist_scale, \
    new_center, out_name, shell_thickness_scale, shell_thickness_min, shell_thickness_max, slice_gap_min=1.5,):
    # cut both outer and inner side of a grain,  give a shell with defined thickness
    # cell_file: str, name of the all-atom model used as an unit cell
    # cg_box_size: [1 x 3] box size of the cg model
    # quat: [w x y z] quaternion for rotation
    # surface_mesh: ovito.data.SurfaceMesh
    # region_id: int, id of the region
    # slice_dist_scale: float, a scaled offset to do slice
    # new_center: [x y z] center of mass
    # out_name: str, name of the output file
    # shell_thickness_scale: float, thickness of the shell relative to center-to-surface distance
    # shell_thickness_min: float, minimum thickness
    # shell_thickness_max: float, maximum thickness
    pipeline = import_file(cell_file)
    ### 1. apply replication
    rep_t = [9, 9, 9]
    pipeline.modifiers.append(ReplicateModifier(num_x=rep_t[0], num_y=rep_t[1],num_z=rep_t[2]))
    ### 2. apply rotation 
    a1 = R.from_quat(quat).as_matrix()
    rotate = AffineTransformationModifier(
          operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[a1[0,0], a1[0,1], a1[0,2], 0],
                            [a1[1,0], a1[1,1], a1[1,2], 0],
                            [a1[2,0], a1[2,1], a1[2,2], 0]],
          only_selected = False)
    pipeline.modifiers.append(rotate)
    ### 3. apply slices
    ##### 3.1. get id of all faces
    face_ids = get_face_id_list_in_region(region_id, surface_mesh)
    ##### 3.2. enumerate face id and apply slice mod
    vertex_coords = surface_mesh.vertices['Position']
    modifiers_list_slice_inner = []
    modifiers_list_slice_outer = []
    for face_id in face_ids:
        p3 = vertex_coords[get_3vertex_id_list_of_face(face_id, surface_mesh)]
        p3[0,:] = closest_point2ref_in_periodic_box(p3[0,:], new_center, cg_box_size)
        p3[1,:] = closest_point2ref_in_periodic_box(p3[1,:], new_center, cg_box_size)
        p3[2,:] = closest_point2ref_in_periodic_box(p3[2,:], new_center, cg_box_size)
        p1 = new_center
        vec, dist = calc_normal_vec(p3, p1)
        relaxed_dist_outer = dist*slice_dist_scale
        relaxed_dist_outer = max(0, min(relaxed_dist_outer, (dist - slice_gap_min)))
        shell_thickness = dist*shell_thickness_scale
        shell_thickness = max(shell_thickness_min, min(shell_thickness, shell_thickness_max))
        relaxed_dist_inner = relaxed_dist_outer - shell_thickness + slice_gap_min
        relaxed_dist_inner = max(0, min(relaxed_dist_inner, relaxed_dist_outer))
        one_slice_inner = SliceModifier(normal=vec, distance=relaxed_dist_inner, select=True)
        one_slice_outer = SliceModifier(normal=vec, distance=relaxed_dist_outer, select=True)
        modifiers_list_slice_inner.append(one_slice_inner)
        modifiers_list_slice_outer.append(one_slice_outer)
    # select particles outside of the inner grain to keep (delete the inverse)
    for modifier_slice in modifiers_list_slice_inner:
        pipeline.modifiers.append(modifier_slice)
        pipeline.modifiers.append(ComputePropertyModifier(
                    output_property = 'is_inner_outer',
                    expressions = '1',
                    only_selected = True                 )
                                 )
    pipeline.modifiers.append(ExpressionSelectionModifier(expression = 'is_inner_outer==1'))
    pipeline.modifiers.append(InvertSelectionModifier())
    pipeline.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Bonded,iterations=3))
    pipeline.modifiers.append(DeleteSelectedModifier())

    # select particles outside the outer grain to delete
    for modifier_slice in modifiers_list_slice_outer:
        pipeline.modifiers.append(modifier_slice)
        pipeline.modifiers.append(ExpandSelectionModifier(mode=ExpandSelectionModifier.ExpansionMode.Bonded,iterations=3))
        pipeline.modifiers.append(DeleteSelectedModifier())
    ### 4. apply move
    move = AffineTransformationModifier(
          operate_on = {'particles'}, # Transform particles but not the box.
          transformation = [[1, 0, 0, new_center[0]],
                            [0, 1, 0, new_center[1]],
                            [0, 0, 1, new_center[2]]],
          only_selected = False)
    pipeline.modifiers.append(move)
    ### 5. change molecule to region_id
    pipeline.modifiers.append(ComputePropertyModifier(
        output_property = 'Molecule Identifier',
        expressions = '%d'%(region_id+1)               )
                             )
    # Export combined dataset to a new file.
    export_file(pipeline, out_name, "lammps/data", atom_style="full")

def modify_shift2vec(frame, data, vec):
    #print(dir(data.particles))
    for i in range(len(data.particles_.positions_)):
        quat = [0, 0, 0, 0]
        quat[0] = data.particles['c_q2'][i]
        quat[1] = data.particles['c_q3'][i]
        quat[2] = data.particles['c_q4'][i]
        quat[3] = data.particles['c_q1'][i]
        rot_mat = R.from_quat(quat).as_matrix()
        vec_rotated = np.matmul(rot_mat, np.array(vec).T)
        data.particles_.positions_[i] += vec_rotated
        data.particles_.identifiers_[i] = i+1
        data.particles_.particle_types_[i] = i+1
        
def merge_lmp_dump_from_list(list_name, out_name):
    data_0 = list_name[0]
    print(data_0)
    pipe_out = import_file(data_0)
    for i in range(1,len(list_name)):
        data_i = list_name[i]
        modifier = CombineDatasetsModifier()
        modifier.source.load(data_i)
        pipe_out.modifiers.append(modifier)
        print(data_i)
    export_file(pipe_out, out_name, "lammps/dump", 
                    columns = ["Particle Identifier", "Particle Type", 
                               "Position.X", "Position.Y", "Position.Z", 
                               "c_q1", "c_q2", "c_q3", "c_q4"])
    return

def add_ghost_supprt_disk(cg_file, rx, ry):
    shifts  = [ [0, 0, 0], [rx/2, 0, 0], [0, ry/2, 0], [-rx/2, 0, 0], [0, -ry/2, 0],
              [rx/4, ry/4, 0], [-rx/4, ry/4, 0], [-rx/4, -ry/4, 0], [rx/4, -ry/4, 0]]
    temp_files = [cg_file + "shift%d.lammpstrj"%(i) for i in range(9)]
    for i, temp_file in enumerate(temp_files):
        pipe_temp = import_file(cg_file)
        modify_temp = functools.partial(modify_shift2vec, vec=shifts[i])
        pipe_temp.modifiers.append(modify_temp)
        #data = pipe_temp.compute()
        #print(data.particles.positions[...])
        export_file(pipe_temp, temp_file, "lammps/dump", 
                    columns = ["Particle Identifier", "Particle Type", 
                               "Position.X", "Position.Y", "Position.Z", 
                               "c_q1", "c_q2", "c_q3", "c_q4"])
    out_file = cg_file + 'supported.lammpstrj'
    merge_lmp_dump_from_list(temp_files, out_file)
    return out_file

def create_parts_aa_ref_cg_voroquat(aa_file, cg_file, vol_cut=-1, wat_file='', slice_dist_scale=0.9, slice_gap_min=1.5):
    pipe_cg = import_file(cg_file)
    pipe_cg.modifiers.append(VoronoiAnalysisModifier(compute_indices = True, generate_polyhedra=True))
    ### remove voronoi with volume larger than vol_cut
    if(vol_cut > 0 and wat_file==''):
        pipe_cg.modifiers.append(ExpressionSelectionModifier(expression = 'Volume>%f'%(vol_cut), operate_on='surface_regions'))
        pipe_cg.modifiers.append(DeleteSelectedModifier())
    data_cg = pipe_cg.compute()
    box_size = [data_cg.cell[0][0], data_cg.cell[1][1], data_cg.cell[2][2]]
    surface = data_cg.surfaces['voronoi-polyhedra']
    particle_ids = surface.regions['Particle Identifier']
    for region_id in range(len(particle_ids)):
        p_index = np.where(data_cg.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        region_vol = surface.regions['Volume'][region_id]
        print(p_index+1, region_vol)
        quat = [0,0,0,0]
        quat[0] = data_cg.particles['voro_quat_x'][p_index]
        quat[1] = data_cg.particles['voro_quat_y'][p_index]
        quat[2] = data_cg.particles['voro_quat_z'][p_index]
        quat[3] = data_cg.particles['voro_quat_w'][p_index]
        pos = data_cg.particles['Position'][p_index]
        out_name = "region_%d.data"%(region_id)
        component_file = aa_file
        if(vol_cut>0 and region_vol>vol_cut and wat_file):
            component_file = wat_file           
        rep_rot_slice_move(component_file, box_size, quat, surface, region_id, slice_dist_scale, pos, out_name, slice_gap_min=slice_gap_min)

def create_shells_aa_ref_cg_voroquat(aa_file, cg_file, vol_cut=-1, wat_file='', slice_dist_scale=0.9, slice_gap_min=1.5, thickness=10):
    pipe_cg = import_file(cg_file)
    pipe_cg.modifiers.append(VoronoiAnalysisModifier(compute_indices = True, generate_polyhedra=True))
    ### remove voronoi with volume larger than vol_cut
    if(vol_cut > 0 and wat_file==''):
        pipe_cg.modifiers.append(ExpressionSelectionModifier(expression = 'Volume>%f'%(vol_cut), operate_on='surface_regions'))
        pipe_cg.modifiers.append(DeleteSelectedModifier())
    data_cg = pipe_cg.compute()
    box_size = [data_cg.cell[0][0], data_cg.cell[1][1], data_cg.cell[2][2]]
    surface = data_cg.surfaces['voronoi-polyhedra']
    particle_ids = surface.regions['Particle Identifier']
    for region_id in range(len(particle_ids)):
        p_index = np.where(data_cg.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        region_vol = surface.regions['Volume'][region_id]
        print(p_index+1, region_vol)
        quat = [0,0,0,0]
        quat[0] = data_cg.particles['voro_quat_x'][p_index]
        quat[1] = data_cg.particles['voro_quat_y'][p_index]
        quat[2] = data_cg.particles['voro_quat_z'][p_index]
        quat[3] = data_cg.particles['voro_quat_w'][p_index]
        pos = data_cg.particles['Position'][p_index]
        out_name = "shell_%d.data"%(region_id)
        component_file = aa_file
        if(vol_cut>0 and region_vol>vol_cut and wat_file):
            component_file = wat_file           
        #rep_rot_slice_move(component_file, box_size, quat, surface, region_id, slice_dist_scale, pos, out_name, slice_gap_min=slice_gap_min)
        rep_rot_slice_shell_move(component_file, box_size, quat, surface, region_id, 
            slice_dist_scale, pos, out_name, shell_thickness_scale=0.1, 
            shell_thickness_min=thickness, shell_thickness_max=thickness, slice_gap_min=slice_gap_min)

def assemble_aa_ref_cg(aa_file, cg_file, vol_cut=-1, wat_file='', slice_dist_scale=0.9, slice_gap_min=1.5):
    pipe_cg = import_file(cg_file)
    pipe_cg.modifiers.append(VoronoiAnalysisModifier(compute_indices = True, generate_polyhedra=True))
    ### remove voronoi with volume larger than vol_cut
    if(vol_cut > 0 and wat_file==''):
        pipe_cg.modifiers.append(ExpressionSelectionModifier(expression = 'Volume>%f'%(vol_cut), operate_on='surface_regions'))
        pipe_cg.modifiers.append(DeleteSelectedModifier())
    data_cg = pipe_cg.compute()
    box_size = [data_cg.cell[0][0], data_cg.cell[1][1], data_cg.cell[2][2]]
    surface = data_cg.surfaces['voronoi-polyhedra']
    particle_ids = surface.regions['Particle Identifier']
    for region_id in range(len(particle_ids)):
        p_index = np.where(data_cg.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        region_vol = surface.regions['Volume'][region_id]
        print(p_index+1, region_vol)
        quat = [0,0,0,0]
        quat[0] = data_cg.particles['c_q2'][p_index]
        quat[1] = data_cg.particles['c_q3'][p_index]
        quat[2] = data_cg.particles['c_q4'][p_index]
        quat[3] = data_cg.particles['c_q1'][p_index]
        pos = data_cg.particles['Position'][p_index]
        out_name = "region_%d.data"%(region_id)
        component_file = aa_file
        if(vol_cut>0 and region_vol>vol_cut and wat_file):
            component_file = wat_file           
        rep_rot_slice_move(component_file, box_size, quat, surface, region_id, slice_dist_scale, pos, out_name, slice_gap_min=slice_gap_min)
        #modifier = CombineDatasetsModifier()
        #modifier.source.load(out_name)
        #pipe_out.modifiers.append(modifier)
    # Export combined dataset to a new file.
    #export_file(pipe_out, cg_file+aa_file, "lammps/data", atom_style="full", ignore_identifiers=True)

def merge_from_list(list_name, out_name):
    data_0 = list_name[0]
    print(data_0)
    pipe_out = import_file(data_0)
    for i in range(1,len(list_name)):
        data_i = list_name[i]
        modifier = CombineDatasetsModifier()
        modifier.source.load(data_i)
        pipe_out.modifiers.append(modifier)
        print(data_i)
    export_file(pipe_out, out_name, "lammps/data", atom_style="full", ignore_identifiers=True)
    return
    
def merge_regions(num_at_each_level):
    file_name_at_each_level = []
    file_list0 = ['region_%d.data'%i for i in range(num_at_each_level[0])]
    file_name_at_each_level.append(file_list0)
    for i,num in enumerate(num_at_each_level[1:]):
        file_list = ['temp_region%d-%d.data'%(i,j) for j in range(num)]
        #print(file_list)
        file_name_at_each_level.append(file_list)
    for i in range(len(file_name_at_each_level)-1):
        list1 = file_name_at_each_level[i]
        list2 = file_name_at_each_level[i+1]
        len1 = len(list1)
        len2 = len(list2)
        temp_merge_list = []
        i2 = 0
        for i1 in range(len1):
            temp_merge_list.append(list1[i1])
            if(len(temp_merge_list)==len1/len2):
                #print(temp_merge_list, list2[i2])
                merge_from_list(temp_merge_list, list2[i2])
                i2 += 1
                temp_merge_list = []
    return file_name_at_each_level[-1][-1]

def merge_shells(num_at_each_level):
    file_name_at_each_level = []
    file_list0 = ['shell_%d.data'%i for i in range(num_at_each_level[0])]
    file_name_at_each_level.append(file_list0)
    for i,num in enumerate(num_at_each_level[1:]):
        file_list = ['temp_shell%d-%d.data'%(i,j) for j in range(num)]
        #print(file_list)
        file_name_at_each_level.append(file_list)
    for i in range(len(file_name_at_each_level)-1):
        list1 = file_name_at_each_level[i]
        list2 = file_name_at_each_level[i+1]
        len1 = len(list1)
        len2 = len(list2)
        temp_merge_list = []
        i2 = 0
        for i1 in range(len1):
            temp_merge_list.append(list1[i1])
            if(len(temp_merge_list)==len1/len2):
                #print(temp_merge_list, list2[i2])
                merge_from_list(temp_merge_list, list2[i2])
                i2 += 1
                temp_merge_list = []
    return file_name_at_each_level[-1][-1]

def refine_box_ff_section(data_file, box_lohi):
    template_box = f'''{box_lohi[0][0]} {box_lohi[0][1]} xlo xhi
{box_lohi[1][0]} {box_lohi[1][1]} ylo yhi
{box_lohi[2][0]} {box_lohi[2][1]} zlo zhi
'''
    template_ff = '''
PairIJ Coeffs # lj/cut/coul/long

1 1 5.03e-06 5.5617 10
1 2 5.03e-06 5.5617 10
1 3 0 2.78085 10
1 4 0 2.78085 10
1 5 0.0008768 4.3636 10
1 6 0.0008701 4.3636 10
1 7 0.0007045 5.4567 10
1 8 1.84e-06 3.302 10
1 9 0.133951 3.215 10
2 2 5.03e-06 5.5617 10
2 3 0 2.78085 10
2 4 0 2.78085 10
2 5 0.0006035 4.4694 10
2 6 0.00146 4.3636 10
2 7 0.00104 4.4425 10
2 8 1.84e-06 3.302 10
2 9 0.133951 3.215 10
3 3 0 0 10
3 4 0 0 10
3 5 0 0 10
3 6 0 0 10
3 7 0 0 10
3 8 0 0 10
3 9 0 1.80852 10
4 4 0 0 10
4 5 0 0 10
4 6 0 0 10
4 7 0 0 10
4 8 0 0 10
4 9 0 1.80852 10
5 5 0.1554 3.1655 10
5 6 0.005253 4.2368 10
5 7 0.8705 2.8965 10
5 8 0.0005302 3.2337 10
5 9 0.151663 3.39128 10
6 6 1.2434 2.7339 10
6 7 0.04554 3.6218 10
6 8 0.0005611 3.271 10
6 9 0.183695 3.23842 10
7 7 0.1554 3.1655 10
7 8 0.0005947 3.263 10
7 9 0.183695 3.23842 10
8 8 1.84e-06 3.302 10
8 9 0.0769592 3.83524 10
9 9 0.148 3.61705 10

Bond Coeffs # harmonic

1 277.067 1
2 277.067 1
3 480 1.4

Angle Coeffs # harmonic

1 90 120
2 22.8848 109.47
'''
    # Read the content of the existing file
    with open(data_file, 'r') as file:
        lines = file.readlines()    
    insert_line_num = 0
    for i, line in enumerate(lines):
        if(line.startswith("Atoms")):
            insert_line_num = i-1
            #print(insert_line_num)
            break
    # Insert the data_string at line 10    
    lines.insert(insert_line_num, template_ff)  
    lines[9:12] = [template_box]  
    #lines.insert(19, template_ff)    

    # Write the modified content back to the file
    with open(data_file, 'w') as file:
        file.writelines(lines)

def discretize_vector(p1, p2, unit_length):
    """
    Discretize a vector from point p1 to point p2 into several unit points.

    Parameters:
        p1 (numpy array): Starting point of the vector (1D NumPy array).
        p2 (numpy array): Ending point of the vector (1D NumPy array).
        unit_length (float): The length of each unit segment.

    Returns:
        numpy array: 2D NumPy array containing discretized unit points.
    """
    # Calculate the vector direction and magnitude
    direction = p2 - p1
    magnitude = np.linalg.norm(direction)

    # If the vector is shorter than the unit length or between 1.2 and 1.5 times the unit length, return p1 and p2
    if magnitude <= 1.5*unit_length:
        return np.array([p1, p2])

    # Calculate the number of unit points to generate
    num_points = round(magnitude / unit_length)

    # Generate the unit points along the vector (excluding p1 and p2)
    unit_points = np.linspace(0, 1, num_points + 1)[1:-1][:, np.newaxis]

    # Calculate the discretized vector points
    discretized_vector = p1 + unit_points * direction

    # Concatenate the original points p1, unit points, and p2
    discretized_points = np.vstack((p1[np.newaxis], discretized_vector, p2[np.newaxis]))

    return discretized_points

def ovito_modify_add_atom_to_bonds(frame, data, unit_length):
    ### topology of bonds [list of tuple particle (index1, index2)] 
    bond_pairs = data.particles.bonds.topology
    new_bond_pairs = []
    all_coords = data.particles["Position"]
    box_size = [data.cell[0][0], data.cell[1][1], data.cell[2][2]]
    for pair in bond_pairs:
        p_index1 = pair[0]
        p_index2 = pair[1]
        coord1 = all_coords[p_index1]
        coord2 = all_coords[p_index2]
        coord2 = closest_point2ref_in_periodic_box(coord2, coord1, box_size)
        ### discretize the bond into several units
        new_particle_coords = discretize_vector(coord1, coord2, unit_length)
        #print( np.linalg.norm(coord2 - coord1) )
        if(len(new_particle_coords)<3):
            new_bond_pairs.append( (p_index1, p_index2) )
            continue
        else:   
            for i, coord in enumerate(new_particle_coords):
                if(i==0 or i==len(new_particle_coords)-1):
                    continue
                data.particles_.add_particle(coord)
                data.particles_.particle_types_[-1] = data.particles['Particle Type'][p_index2]
                data.particles_.identifiers_[-1] = data.particles.count
                data.particles_['Molecule Identifier'][-1] = data.particles['Molecule Identifier'][p_index2]
                if(i==1):
                    new_bond_pairs.append( (p_index1, data.particles.count-1) )
                if(i>1):
                    new_bond_pairs.append( (data.particles.count-2, data.particles.count-1) )
                if(i==len(new_particle_coords)-2):
                    new_bond_pairs.append( (data.particles.count-1, p_index2) )
    #data.particles_.bonds_ = None
    print("created %d new bonds"%(len(new_bond_pairs)))
    bonds = data.particles_.create_bonds( count=len(new_bond_pairs) )
    bonds.create_property('Topology', data=new_bond_pairs)
    bonds.bond_types_[:] = 1

def ovito_modify_add_particle(frame, data, position):
    data.particles_.add_particle(position)
    
def ovito_modify_assign_type(frame, data, particle_index, property_val):
    # use type as a placeholder
    data.particles_.particle_types_[particle_index] = property_val
    
def generate_points_on_voro_cell_face_center(frame, data):
    box_size = [data.cell[0][0], data.cell[1][1], data.cell[2][2]]
    surface = data.surfaces['voronoi-polyhedra']
    vertex_coords = surface.vertices['Position'] 
    particle_ids = surface.regions['Particle Identifier']
    ### find all dual faces ###
    stacked_face_ids = []
    stacked_vertex_ids = []
    dual_face_id_pairs = {}
    for region_id in range(len(particle_ids)):
        all_face_ids = get_face_id_list_in_region(region_id, surface)
        for face_id in all_face_ids:
            vertex_ids = tuple( np.sort(get_all_vertices(face_id, surface)) )
            #print(vertex_ids)
            #print(stacked_vertex_ids)
            #if( not all((vertices==record).any() for record in stacked_vertices) ):
            if( not vertex_ids in stacked_vertex_ids ):
                stacked_vertex_ids.append(vertex_ids)
                stacked_face_ids.append(face_id)
            else:
                stacked_index = stacked_vertex_ids.index(vertex_ids)
                dual_face_id = stacked_face_ids[stacked_index]
                dual_face_id_pairs[face_id] = dual_face_id
                dual_face_id_pairs[dual_face_id] = face_id
                stacked_face_ids.pop(stacked_index)
                stacked_vertex_ids.pop(stacked_index)
    print(len(dual_face_id_pairs))
    bond_pairs = []
    bond_types = []
    point_added_face_ids = []
    point_id_offset = data.particles.count
    for region_id in range(len(particle_ids)):
        p_index = np.where(data.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        if(p_index%100==0):
            print(p_index)
        center_pos = data.particles['Position'][p_index]
        all_face_ids = get_face_id_list_in_region(region_id, surface)
        all_face_area = []
        all_face_vertices = []
        for face_id in all_face_ids:
            vertices = vertex_coords[get_all_vertices(face_id, surface)]
            vertices_close = np.array([closest_point2ref_in_periodic_box(v, center_pos, box_size) for v in vertices])
            pyny.Polygon.verify = False
            face_area = pyny.Polygon(vertices_close).get_area()
            all_face_area.append(face_area)
            all_face_vertices.append(vertices_close)
        all_face_area = np.array(all_face_area)
        face_area_top2 = all_face_area.argsort()[-2:]
        face1_index = face_area_top2[0]
        face2_index = face_area_top2[1]
        face1_id = all_face_ids[face1_index]
        face2_id = all_face_ids[face2_index]
        if(not face1_id in point_added_face_ids and not dual_face_id_pairs[face1_id] in point_added_face_ids):
            face1_center = np.average(all_face_vertices[face1_index], axis=0)
            data.particles_.add_particle(face1_center)
            data.particles_.particle_types_[-1] = data.particles['Particle Identifier'][p_index]
            data.particles_.identifiers_[-1] = data.particles.count
            point_added_face_ids.append(face1_id)
        if(not face2_id in point_added_face_ids and not dual_face_id_pairs[face2_id] in point_added_face_ids):
            face2_center = np.average(all_face_vertices[face2_index], axis=0)
            data.particles_.add_particle(face2_center)
            data.particles_.particle_types_[-1] = data.particles['Particle Identifier'][p_index]
            data.particles_.identifiers_[-1] = data.particles.count
            point_added_face_ids.append(face2_id)
    ### create molecule identifier to store the voro cell index for each particle ###
    mol_ids = data.particles_.create_property('Molecule Identifier')
    mol_ids[:] = data.particles.particle_types[:]
    mol_ids[0:point_id_offset] = data.particles['Particle Identifier'][0:point_id_offset]
    data.particles_.particle_types_[:] = 2
    data.particles_.particle_types_[0:point_id_offset] = 1
    ### connect the atoms
    for region_id in range(len(particle_ids)):
        p_index = np.where(data.particles['Particle Identifier']==particle_ids[region_id])[0][0]
        all_face_ids = get_face_id_list_in_region(region_id, surface)
        face_p_indices = []
        for face_id in all_face_ids:
            ### collect all faces with a particle added
            face_p_index = -1
            if(face_id in point_added_face_ids):
                face_p_index = point_added_face_ids.index(face_id) + point_id_offset
            elif(dual_face_id_pairs[face_id] in point_added_face_ids):
                face_p_index = point_added_face_ids.index(dual_face_id_pairs[face_id]) + point_id_offset
            if(face_p_index>0):
                face_p_indices.append(face_p_index)
        for face_p_index in face_p_indices:
            bond_pairs.append((p_index, face_p_index))
            bond_types.append(1)
        
    bonds = data.particles_.create_bonds( count=len(bond_pairs) )
    bonds.create_property('Topology', data=bond_pairs)
    bonds.create_property('Bond Type', data=bond_types)
    ### use charge as a holder for degree of the node (atom)
    graph = nx.Graph()
    graph.add_edges_from(bond_pairs)
    charges = data.particles_.create_property('Charge')
    for node, degree in graph.degree():
        charges[node] = degree

def discretize_vector(p1, p2, unit_length):
    """
    Discretize a vector from point p1 to point p2 into several unit points.

    Parameters:
        p1 (numpy array): Starting point of the vector (1D NumPy array).
        p2 (numpy array): Ending point of the vector (1D NumPy array).
        unit_length (float): The length of each unit segment.

    Returns:
        numpy array: 2D NumPy array containing discretized unit points.
    """
    # Calculate the vector direction and magnitude
    direction = p2 - p1
    magnitude = np.linalg.norm(direction)

    # If the vector is shorter than the unit length or between 1.2 and 1.5 times the unit length, return p1 and p2
    if magnitude <= 1.5*unit_length:
        return np.array([p1, p2])

    # Calculate the number of unit points to generate
    num_points = round(magnitude / unit_length)

    # Generate the unit points along the vector (excluding p1 and p2)
    unit_points = np.linspace(0, 1, num_points + 1)[1:-1][:, np.newaxis]

    # Calculate the discretized vector points
    discretized_vector = p1 + unit_points * direction

    # Concatenate the original points p1, unit points, and p2
    discretized_points = np.vstack((p1[np.newaxis], discretized_vector, p2[np.newaxis]))

    return discretized_points

def find_removable_edge_keep_connected(graph, edge_list, length_bound=10):
    """
    Find an edge from the edge_list that can be removed without affecting the connectivity of the graph.

    Parameters:
        graph (NetworkX graph): The input graph.
        edge_list (list): List of edges to consider for removal.

    Returns:
        tuple: A tuple containing the removable edge as (u, v) where u and v are nodes of the graph.
               If no removable edge is found, return None.
    """
    # Check if the graph is connected
    if not nx.is_connected(graph):
        return None
        #raise ValueError("The graph is not connected.")

    # Find all simple cycles in the graph
    cycles = nx.simple_cycles(graph, length_bound=length_bound)

    # Sort the cycles based on their length (number of nodes in the cycle)
    sorted_cycles = sorted(cycles, key=lambda cycle: len(cycle))
    #print(len(sorted_cycles[0]))
    #sorted_cycles = sorted(cycles)

    # Check if any edge in the edge_list appears in a cycle
    for cycle in sorted_cycles:
        for edge in edge_list:
            if edge in zip(cycle, cycle[1:] + [cycle[0]]):
                # Edge found in a cycle, it is safe to remove it
                return edge
            if (edge[1], edge[0]) in zip(cycle, cycle[1:] + [cycle[0]]):
                return (edge[1], edge[0])

    # No removable edge found
    return None

def find_removable_node_degree2_with_degree3_neighbors(G):
    # Check if the graph is connected
    if not nx.is_connected(G):
        return None  # Return None if the graph is not connected
        # Alternatively, you can raise a ValueError to indicate that the graph is not connected
        # raise ValueError("The graph is not connected.")
        
    node_to_remove = None
    for node in G.nodes():
        # Check if the node's degree is 2
        if G.degree[node] == 2:
            # Get the list of neighbors for the current node
            neighbors = list(G.neighbors(node))
            
            # Check if both neighbors have a degree of 3
            if G.degree[neighbors[0]] == 3 and G.degree[neighbors[1]] == 3:
                # If both neighbors have a degree of 3, set the node_to_remove to the current node and break the loop
                node_to_remove = [node]
                break
    
    return node_to_remove  # Return the node to remove (or None if no suitable node is found)

def find_removable_node_degree2_on_degree3_path(G, path_length):
    # Step 1: Find all degree-3 nodes
    degree_3_nodes = [node for node in G.nodes() if G.degree[node] == 3]

    # Step 2: Construct all pairs from the degree-3 nodes
    node_pairs = [(u, v) for u in degree_3_nodes for v in degree_3_nodes if u != v]

    # Step 3 and 4: Find the first path whose length is shorter than path_length
    nodes_on_path = []
    for source, target in node_pairs:
        for path in nx.all_simple_paths(G, source=source, target=target, cutoff=path_length):
            if len(path) == path_length:
                nodes_on_path = path[1:-1]
                break
        if nodes_on_path:
            break

    return nodes_on_path  # Return the list of nodes on the path, or an empty list if no such path is found

def ovito_modify_select_particles(frame, data, particle_index_list):
    selection = data.particles_.create_property('Selection')
    selection[particle_index_list] =  1

def ovito_modify_select_bonds(frame, data, bond_pair_list):
    all_bond_pairs = data.particles.bonds.topology
    selection = data.particles_.bonds_.create_property('Selection')
    for i in range(len(selection)):
        pair = all_bond_pairs[i]
        pair = ( pair[0], pair[1] )
        dual = ( pair[1], pair[0] )
        if( pair in bond_pair_list or dual in bond_pair_list ):
            selection[i] = 1

def ovito_modify_mark_degree_to_charge(frame, data):
    bond_pairs = data.particles.bonds.topology
    ### use charge as a holder for degree of the node (atom)
    graph = nx.Graph()
    graph.add_edges_from(bond_pairs)
    charges = data.particles_.create_property('Charge')
    for node, degree in graph.degree():
        charges[node] = degree

def cut_degree3_nodes_from_net(in_file, out_file, remove_ratio=1.0):
    pipeline = import_file(in_file)
    data = pipeline.compute()
    bond_pairs = data.particles.bonds.topology
    bond_types = data.particles.bonds['Bond Type']

    graph = nx.Graph()
    graph.add_edges_from(bond_pairs)

    particles_to_remove = []
    bonds_to_remove = []

    while(nx.is_connected(graph)):
        nodes = find_removable_node_degree2_with_degree3_neighbors(graph)
        temp = graph.copy()
        if nodes:
            temp.remove_nodes_from(nodes)
        if(nodes and nx.is_connected(temp)):
            particles_to_remove.extend(nodes)
            graph = temp
        else:
            break

    while(nx.is_connected(graph)):
        path_len = 4
        nodes = find_removable_node_degree2_on_degree3_path(graph, path_len)
        while not nodes:
            path_len += 1
            nodes = find_removable_node_degree2_on_degree3_path(graph, path_len)
            if(path_len > 100):
                break
        temp = graph.copy()
        temp.remove_nodes_from(nodes)
        if(nodes and nx.is_connected(temp)):
            particles_to_remove.extend(nodes)
            graph = temp
        else:
            break
            
    # Find all degree-3 nodes
    degree_3_nodes = [node for node, degree in graph.degree() if degree == 3]
    # Find all edges attached to degree-3 nodes
    edges_attached_to_degree_3_nodes = []
    for node in degree_3_nodes:
        edges_attached_to_degree_3_nodes.extend(list(graph.edges(node)))
        
    type3bonds = edges_attached_to_degree_3_nodes
    edge = 1
    count = 0
    while edge:
        if(edge==1):
            edge = find_removable_edge_keep_connected(graph, type3bonds, length_bound=100 )
            continue
        count += 1
        graph.remove_edge(edge[0],edge[1])
        bonds_to_remove.append( edge )
        edge = find_removable_edge_keep_connected(graph, type3bonds, length_bound=100 )

    print("maximum node to remove: %d"%len(particles_to_remove))
    print("maximum edge to remove: %d"%len(bonds_to_remove))
    max_num_to_remove = len(particles_to_remove)*2 + len(bonds_to_remove)
    tol_num_to_remove = round(max_num_to_remove*remove_ratio)
    if(tol_num_to_remove <= len(particles_to_remove)*2):
        particles_to_remove = particles_to_remove[0:round(tol_num_to_remove/2)]
        bonds_to_remove = []
    elif(tol_num_to_remove < max_num_to_remove):
        #particles_to_remove = particles_to_remove
        bonds_to_remove = bonds_to_remove[0:(tol_num_to_remove-len(particles_to_remove)*2)]

    modify = functools.partial(ovito_modify_select_particles, particle_index_list=particles_to_remove)
    pipeline.modifiers.append(modify)
    modify = functools.partial(ovito_modify_select_bonds, bond_pair_list=bonds_to_remove)
    pipeline.modifiers.append(modify)
    pipeline.modifiers.append( DeleteSelectedModifier() )
    pipeline.modifiers.append( ovito_modify_mark_degree_to_charge ) 

    export_file(pipeline, out_file, "lammps/data", atom_style="full", ignore_identifiers=True)

def ovito_modifier_reset_molecule_id_from_1_to_n(frame, data):
    old_mol_ids = data.particles['Molecule Identifier']
    seq_mod_ids = np.unique(old_mol_ids)
    new_mol_ids = data.particles_.create_property('Molecule Identifier')
    
    for i, mol_id in enumerate(seq_mod_ids):
        indices = np.where(old_mol_ids==mol_id)[0]
        new_mol_ids[indices] = i+1

def reset_molecule_id_from_1_to_n(file_in, file_out):   
    pipeline = import_file(file_in)
    data = pipeline.compute()
    box_3d_lo_hi = np.zeros((3,2))
    box_3d_lo_hi[:,0] = data.cell[:,3]
    box_3d_lo_hi[:,1] = data.cell[:,3] + [data.cell[0,0], data.cell[1,1], data.cell[2,2]]
    pipeline.modifiers.append(ovito_modifier_reset_molecule_id_from_1_to_n)
    export_file(pipeline, file_out, "lammps/data", atom_style="full", ignore_identifiers=True)
    refine_box_ff_section(file_out, box_3d_lo_hi)
    return
