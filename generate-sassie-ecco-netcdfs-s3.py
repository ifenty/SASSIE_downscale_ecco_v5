#!/usr/bin/env python
# coding: utf-8

# # Process SASSIE ocean model granules on s3

## import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc4
import tarfile
import json
import uuid as uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
import s3fs
import argparse

## import ECCO utils
import sys
sys.path.append('/Users/mzahn/github_others/ECCOv4-py')
import ecco_v4_py as ecco


## Define functions

def load_sassie_N1_field(file_dir, fname, nk=1, skip=0):
    num_cols = 680*4 + 1080
    num_rows = 1080
    
    time_level = int(fname.split('.data')[0].split('.')[-1])
    
    tmp_compact = ecco.load_binary_array(file_dir, fname, \
                                    num_rows, num_cols, nk=nk, skip=skip, filetype='>f4')

    return tmp_compact, time_level


def sassie_n1_compact_to_faces_2D(sassie_n1_compact):
    sassie_faces = dict()
    n = 680
    
    # Face 1 
    start_row = 0
    end_row = n
    sassie_faces[1] = sassie_n1_compact[start_row:end_row,:]

    # Face 2
    start_row = end_row
    end_row = start_row + n

    sassie_faces[2] = sassie_n1_compact[start_row:end_row,:]
    
    # Face 3
    start_row = end_row
    end_row = start_row + 1080
    sassie_faces[3] = sassie_n1_compact[start_row:end_row:,:]
    
    #Face 4
    start_row = end_row
    end_row = end_row + 680
    sassie_faces[4] = sassie_n1_compact[start_row:end_row].reshape(1080, n)

    #Face 5
    start_row = end_row
    end_row = end_row + 680
    sassie_faces[5] = sassie_n1_compact[start_row:end_row].reshape(1080, n)

    return sassie_faces


def sassie_n1_compact_to_faces_3D(sassie_n1_compact):
    sassie_faces = dict()
    n = 680
    
    # Face 1 
    start_row = 0
    end_row = n
    sassie_faces[1] = sassie_n1_compact[:,start_row:end_row,:]

    # Face 2
    start_row = end_row
    end_row = start_row + n
    sassie_faces[2] = sassie_n1_compact[:,start_row:end_row,:]
    
    # Face 3
    start_row = end_row
    end_row = start_row + 1080
    sassie_faces[3] = sassie_n1_compact[:,start_row:end_row:,:]
    
    #Face 4
    start_row = end_row
    end_row = end_row + 680
    sassie_faces[4] = sassie_n1_compact[:,start_row:end_row].reshape(90, 1080, n)

    #Face 5
    start_row = end_row
    end_row = end_row + 680
    sassie_faces[5] = sassie_n1_compact[:,start_row:end_row].reshape(90, 1080, n)

    return sassie_faces


def combine_sassie_N1_faces_to_HHv2_2D(face_arr):
    """
    2D function for scalar fields, c point
    """
    # dimensions of the final Arctic HH field. 535+185+1080=1800
    new_arr = np.zeros((1080, 1800)) 
    
    # cut out sections we want and assign them to location on HH
    new_arr[:, 185:185 + 1080] = face_arr[3]
    
    # rotate Face 1 to line up with orientation of Face 3
    new_arr[:, 0:185] = np.flipud(face_arr[1][-185:,:].T) # flip and transpose
    
    new_arr[:, 185 + 1080:] = face_arr[4][:,:535]

    new_arr = np.rot90(new_arr,2) # rotate it 180 so Greenland/AK are on bottom
    return new_arr


def combine_sassie_N1_faces_to_HHv2_2D_u_point(face_arr_u, face_arr_v):
    """
    2D function for vector fields, u point
    """
    ## dimensions of the final Arctic HH field. 535+185+1080=1800
    new_arr = np.zeros((1080, 1800))
    
    ## add Arctic face (3)
    new_arr[:, 185:185+1080] = face_arr_u[3] # take entire Artic face
    
    ## add face 1 that will be flipped (must use v array)
    new_arr[:, 0:185] = np.flipud(face_arr_v[1][-185:,:].T)
        
    ## add part of face 4 (Alaska)
    new_arr[:, 185+1080:] = face_arr_u[4][:,:535]

    ## rotate by 90 deg twice to have Alaska on bottom left
    ## since it is vector field, have to multiply whole array by -1
    new_arr = np.rot90(new_arr,2)
    new_arr = new_arr *-1
        
    return new_arr


def combine_sassie_N1_faces_to_HHv2_2D_v_point(face_arr_v, face_arr_u, vec=False):
    """
    2D function for vector fields, v point
    """
    ## dimensions of the final Arctic HH field. 535+185+1080=1800
    new_arr = np.zeros((1080, 1800))
    
    ## add Arctic face (3)
    new_arr[:, 185:185+1080] = face_arr_v[3]
    
    ## add part of face 1 (Europe) that will be flipped (must use u array and multiply by -1)
    
    ## after rotating face 1, the u points on face 1 will not match the v points of face 3 (offset by 1 upwards)
    ## therefore, must remove the first column of face 1 and add the first column from face 2 to the end of face 1
    ## remove the first i column from the u field so the shape is (680, 1079) = (j,i)
    face1_tmp = face_arr_u[1][:,1:]
    ## then add the first row from face 2 to the end of face 1
    face1_mod = np.concatenate((face1_tmp, face_arr_u[2][:,:1]), axis=1)
    
    ## add modified face 1 by rotating and multiplying by -1
    new_arr[:, 0:185] = np.flipud(face1_mod[-185:,:].T)*-1
    
    ## add part of face 4 (Alaska)
    new_arr[:, 185+1080:] = face_arr_v[4][:,:535]

    ## rotate by 90 deg twice to have Alaska on bottom left
    new_arr = np.rot90(new_arr,2)
    new_arr = new_arr *-1
        
    return new_arr


def combine_sassie_N1_faces_to_HHv2_3D(face_arr):
    """
    3D function for scalar fields, c point
    """
    # dimensions of the final Arctic HH field. 535+185+1080=1800 ; 90 vertical levels
    new_arr = np.zeros((90, 1080, 1800)) 
    
    # cut out sections we want and assign them to location on HH
    new_arr[:,:,185:185+1080] = face_arr[3]
    
    # rotate Face 1 to line up with orientation of Face 3
    new_arr[:,:,0:185] = np.transpose(face_arr[1][:,-185:,::-1],axes=(0,2,1)) # flip and transpose
    
    ## add part of face 4 (Alaska)
    new_arr[:,:,185+1080:] = face_arr[4][:,:,:535]
    
    ## rotate it 180 so Greenland/AK are on bottom
    new_arr = np.rot90(new_arr,2,axes=(1,2)) 
    
    return new_arr


def combine_sassie_N1_faces_to_HHv2_3D_u_point(face_arr_u, face_arr_v):
    """
    3D function for vector fields, u point
    """
    ## dimensions of the final Arctic HH field. 535+185+1080=1800 ; 90 vertical levels
    new_arr = np.zeros((90, 1080, 1800))
    
    ## add Arctic face (3)
    new_arr[:,:,185:185+1080] = face_arr_u[3] # take entire Artic face
    
    ## add face 1 that will be flipped (must use v array)
    new_arr[:,:,0:185] = np.transpose(face_arr_v[1][:,-185:,::-1],axes=(0,2,1))
        
    ## add part of face 4 (Alaska)
    new_arr[:,:,185+1080:] = face_arr_u[4][:,:,:535]

    ## rotate by 90 deg twice to have Alaska on bottom left
    ## since it is vector field, have to multiply whole array by -1
    new_arr = np.rot90(new_arr,2,axes=(1,2))
    new_arr = new_arr *-1
        
    return new_arr


def combine_sassie_N1_faces_to_HHv2_3D_v_point(face_arr_v, face_arr_u):
    """
    3D function for vector fields, v point
    """
    ## dimensions of the final Arctic HH field. 535+185+1080=1800
    new_arr = np.zeros((90, 1080, 1800))
    
    ## add Arctic face (3)
    new_arr[:,:,185:185+1080] = face_arr_v[3]
    
    ## add part of face 1 (Europe) that will be flipped (must use u array and multiply by -1)
    
    ## after rotating face 1, the u points on face 1 will not match the v points of face 3 (offset by 1 upwards)
    ## therefore, must remove the first column of face 1 and add the first column from face 2 to the end of face 1
    ## remove the first i column from the u field so the shape is (680, 1079) = (j,i)
    face1_tmp = face_arr_u[1][:,:,1:]
    ## then add the first row from face 2 to the end of face 1
    face1_mod = np.concatenate((face1_tmp, face_arr_u[2][:,:,:1]), axis=2)
    
    ## add modified face 1 by rotating and multiplying by -1
    new_arr[:,:,0:185] = np.transpose(face1_mod[:,-185:,::-1],axes=(0,2,1))*-1
    
    ## add part of face 4 (Alaska)
    new_arr[:,:,185+1080:] = face_arr_v[4][:,:,:535]

    ## rotate by 90 deg twice to have Alaska on bottom left
    new_arr = np.rot90(new_arr,2,axes=(1,2))
    new_arr = new_arr *-1
        
    return new_arr


def timestamp_from_iter_num(iter_num):
    """
    takes the model iteration that was pulled from the data's filename and converts it to its equivalent datetime
    """
    ## Start time of the model is 5790000 (22.0319 years after 1992-01-01)
    ## there are 120 seconds for each iteration and 86400 seconds per day
    ## take the iteration number, convert to seconds, and calculate number of days since start of model
    
    ## from Mike: "Near the end of the simulation, I ran into some sort of instability so I changed the time step from 120 seconds to 60 seconds.
    ## Usually I would change it back to 120 second after getting past the instability but I was kinda close to the end so I just let it ride with 60 seconds."
    if iter_num > 1e7:
        iter_num = iter_num/2
    
    num_days_since_start = iter_num*120 / 86400 ## divide iter_number by 86400 which is equal to the number of seconds in a day
    
    model_start_time = datetime(1992,1,1) # data.cal start time is 1992-01-01
    timestamp = np.array([model_start_time + timedelta(days=num_days_since_start)], dtype='datetime64[ns]')
    
    return timestamp


def unpack_tar_gz_files(data_dir):
    ## see if tar.gz files were already decompressed
    data_files = list(data_dir.glob('*.data'))
    if len(data_files)>0:
        print("tar.gz files already unpacked")
    ## if not, open them
    else:
        ## pull list of all tar.gz files in directory
        tar_gz_files = list(data_dir.glob('*.tar.gz'))
        
        ## unzip targz file
        for file_path in tar_gz_files:
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(data_dir) # save files to same directory
            tar.close()


def make_2D_HHv2_ds(field_HH, model_grid_ds, timestamp, grid_point, da_name):
    
    ## get time bounds and center time
    time_bnds, center_time = ecco.make_time_bounds_from_ds64(timestamp[0], 'AVG_DAY')
    time_bnds_da = xr.DataArray(time_bnds.reshape(1,2), dims=['time', 'nv'])
    
    ## create DataArray for c point data
    if grid_point == 'c':
        tmp_da = xr.DataArray([field_HH], dims=['time','j','i'],\
                                coords={'time':(('time'), np.array([center_time]))})
    
    ## create DataArray for u point data
    elif grid_point == 'u':
        tmp_da = xr.DataArray([field_HH], dims=['time','j','i_g'],\
                                coords={'time':(('time'), np.array([center_time]))})
    
    ## create DataArray for v point data
    elif grid_point == 'v':
        tmp_da = xr.DataArray([field_HH], dims=['time','j_g','i'],\
                                coords={'time':(('time'), np.array([center_time]))})
    
    ## name the array
    tmp_da.name = da_name
    
    ## add additional coordinates to dataset
    tmp_ds = tmp_da.to_dataset().assign_coords({
        'time_bnds':time_bnds_da,\
        'XC':model_grid_ds.XC,\
        'YC':model_grid_ds.YC,\
        'XG':model_grid_ds.XG,\
        'YG':model_grid_ds.YG,\
        'XC_bnds':model_grid_ds.XC_bnds,\
        'YC_bnds':model_grid_ds.YC_bnds,\
        'Zp1':model_grid_ds.Zp1})
    
    return tmp_ds


def make_3D_HHv2_ds(field_HH, model_grid_ds, timestamp, grid_point, da_name, k_face='center'):
    
    ## get time bounds and center time
    time_bnds, center_time = ecco.make_time_bounds_from_ds64(timestamp[0], 'AVG_DAY')
    time_bnds_da = xr.DataArray(time_bnds.reshape(1,2), dims=['time', 'nv'])
    
    if k_face == 'center':
        
        ## create DataArray for c point data, center
        if grid_point == 'c':
            tmp_da = xr.DataArray([field_HH], dims=['time', 'k','j','i'],\
                                    coords={'time':(('time'),np.array([center_time]))})
            tmp_da['k'].attrs['axis']  = 'Z'
            
        ## create DataArray for u point data, center
        elif grid_point == 'u':
            tmp_da = xr.DataArray([field_HH], dims=['time', 'k','j','i_g'],\
                                    coords={'time':(('time'),np.array([center_time]))})
            tmp_da['k'].attrs['axis']  = 'Z'
            
        ## create DataArray for v point data, center
        elif grid_point == 'v':
            tmp_da = xr.DataArray([field_HH], dims=['time', 'k','j_g','i'],\
                                    coords={'time':(('time'),np.array([center_time]))})
            tmp_da['k'].attrs['axis']  = 'Z'            
            
    elif k_face == 'top':
        ## create DataArray for c point data, top
        tmp_da = xr.DataArray([field_HH], dims=['time','k_l','j','i'],\
                                coords={'time':(('time'),np.array([center_time]))})
        tmp_da['k_l'].attrs['axis']  = 'Z'

    ## name the array
    tmp_da.name = da_name
        
    ## add additional coordinates to dataset
    tmp_ds = tmp_da.to_dataset().assign_coords({
        'time_bnds':time_bnds_da,\
        'XC':model_grid_ds.XC,\
        'YC':model_grid_ds.YC,\
        'XG':model_grid_ds.XG,\
        'YG':model_grid_ds.YG,\
        'XC_bnds':model_grid_ds.XC_bnds,\
        'YC_bnds':model_grid_ds.YC_bnds,\
        'Z':model_grid_ds.Z,\
        'Zu':model_grid_ds.Zu,\
        'Zl':model_grid_ds.Zl,\
        'Zp1':model_grid_ds.Zp1})
        
 #   tmp_da = add_geo_metadata(tmp_da)

    return tmp_ds


def process_2D_variable(data_dir, filename, var_tmp_table, vars_table, sassie_n1_geometry_ds):
    
    var_name = var_tmp_table['variable'].values[0]
    n_skip = var_tmp_table['field_index'].values[0] * 1
    grid_point = var_tmp_table.cgrid_point.values
    
    ## process binary data to compact format
    data_compact, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=1, skip=n_skip)
    
    ## convert compact format to 5 faces
    data_faces = sassie_n1_compact_to_faces_2D(data_compact)
    
    ## convert faces to HHv2 Arctic rectangle
    ## data on u and v points need to be handled differently from c points
    if var_tmp_table.data_type.values == 'V': # if it is a vector field
        var_mate = var_tmp_table.mate.values[0]
        var_table_mate = vars_table[vars_table.variable.values == var_mate]
        var_mate_field_index = var_table_mate.field_index.values[0]
        
        if grid_point == 'v':
            ## get u field
            n_skip_u = var_mate_field_index
            data_compact_u, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=1, skip=n_skip_u)
            face_arr_u = sassie_n1_compact_to_faces_2D(data_compact_u)
            
            ## process v field
            var_HHv2 = combine_sassie_N1_faces_to_HHv2_2D_v_point(data_faces, face_arr_u)
            
        elif grid_point == 'u':
            ## get v field
            n_skip_v = var_mate_field_index
            data_compact_v, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=1, skip=n_skip_v)
            face_arr_v = sassie_n1_compact_to_faces_2D(data_compact_v)
            
            ## process u field
            var_HHv2 = combine_sassie_N1_faces_to_HHv2_2D_u_point(data_faces, face_arr_v)
            
    elif var_tmp_table.data_type.values == 'S': # if it is a scalar field
        var_HHv2 = combine_sassie_N1_faces_to_HHv2_2D(data_faces)
    
    ## add timestamp adn create dataset
    timestamp = timestamp_from_iter_num(iter_num)
    var_HHv2_ds = make_2D_HHv2_ds(var_HHv2, sassie_n1_geometry_ds, timestamp, grid_point=grid_point, da_name=var_name)
    
    return var_HHv2_ds


def process_3D_variable(data_dir, filename, var_tmp_table, vars_table, sassie_n1_geometry_ds):
    
    var_name = var_tmp_table['variable'].values[0]
    
    ## there are 90 vertical levels; use index from table to identify how many fields to skip
    n_skip = var_tmp_table['field_index'].values[0] * 90
    var_k_face = var_tmp_table['k_face'].values[0]
    grid_point = var_tmp_table.cgrid_point.values
    
    ## process binary data to compact format
    data_compact, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=90, skip=n_skip)
    
    ## convert compact format to 5 faces
    data_faces = sassie_n1_compact_to_faces_3D(data_compact)
    
    ## convert faces to HHv2 Arctic rectangle
    ## data on u and v points need to be handled differently from c points
    if var_tmp_table.data_type.values == 'V': # if it is a vector field
        var_mate = var_tmp_table.mate.values[0]
        var_table_mate = vars_table[vars_table.variable.values == var_mate]
        var_mate_field_index = var_table_mate.field_index.values[0]
        
        if grid_point == 'v':
            ## get u field
            n_skip_u = var_mate_field_index * 90
            data_compact_u, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=90, skip=n_skip_u)
            face_arr_u = sassie_n1_compact_to_faces_3D(data_compact_u)
            
            ## process v field
            var_HHv2 = combine_sassie_N1_faces_to_HHv2_3D_v_point(data_faces, face_arr_u)
            
        elif grid_point == 'u':
            ## get v field
            n_skip_v = var_mate_field_index * 90
            data_compact_v, iter_num = load_sassie_N1_field(str(data_dir), filename, nk=90, skip=n_skip_v)
            face_arr_v = sassie_n1_compact_to_faces_3D(data_compact_v)
            
            ## process u field
            var_HHv2 = combine_sassie_N1_faces_to_HHv2_3D_u_point(data_faces, face_arr_v)
        
    elif var_tmp_table.data_type.values == 'S': # if it is a scalar field
        var_HHv2 = combine_sassie_N1_faces_to_HHv2_3D(data_faces) # c point
    
    ## add timestamp and create dataset
    timestamp = timestamp_from_iter_num(iter_num)
    var_HHv2_ds = make_3D_HHv2_ds(var_HHv2, sassie_n1_geometry_ds, timestamp, grid_point=grid_point, da_name=var_name, k_face=var_k_face)
    
    return var_HHv2_ds


def mask_dry_grid_cells(ds, var, geometry_ds, grid_point):
    ## make copy of dataset
    ds_tmp = ds.copy(deep=True)
    
    ## tracer points use maskC, u points use maskW, and v points use maskS
    if grid_point == 'c':
        ds_tmp[var] = ds[var].where(geometry_ds.maskC==True)
        
    elif grid_point == 'v':
        ds_tmp[var] = ds[var].where(geometry_ds.maskS==True)
        
    elif grid_point == 'u':
        ds_tmp[var] = ds[var].where(geometry_ds.maskW==True)
    
    return ds_tmp


def create_encoding(ecco_ds, output_array_precision = np.float32):
    
    # Create NetCDF encoding directives
    # ---------------------------------------------
    # print('\n... creating variable encodings')
    # ... data variable encoding directives
    
    # Define fill values for NaN
    if output_array_precision == np.float32:
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif output_array_precision == np.float64:
        netcdf_fill_value = nc4.default_fillvals['f8']
    
    dv_encoding = dict()
    for dv in ecco_ds.data_vars:
        dv_encoding[dv] =  {'zlib':True, \
                            'complevel':5,\
                            'shuffle':True,\
                            '_FillValue':netcdf_fill_value}

    # ... coordinate encoding directives
    # print('\n... creating coordinate encodings')
    coord_encoding = dict()
    for coord in ecco_ds.coords:
        # set default no fill value for coordinate
        if output_array_precision == np.float32:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
        elif output_array_precision == np.float64:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float64'}

        # force 64 bit ints to be 32 bit ints
        if (ecco_ds[coord].values.dtype == np.int32) or \
           (ecco_ds[coord].values.dtype == np.int64) :
            coord_encoding[coord]['dtype'] ='int32'

        # fix encoding of time
        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in ecco_ds[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = ecco_ds[coord].attrs['units']
                # delete from the attributes list
                del ecco_ds[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # ... combined data variable and coordinate encoding directives
    encoding = {**dv_encoding, **coord_encoding}

    return encoding


def modify_metadata(ds, var, var_filename_netcdf):   
    title = 'SASSIE Ocean Model ' + var + ' Parameter for the Lat-Lon-Cap 1080 (llc1080) Native Model Grid (Version 1 Release 1)'
    
    ## edit specific metadata for these datasets
    ds.attrs['author'] = 'Mike Wood, Marie Zahn, and Ian Fenty'
    ds.attrs['comment'] = 'SASSIE llc1080 V1R1 fields are consolidated onto a single curvilinear grid face focusing on the Arctic domain using fields from the 5 faces of the lat-lon-cap 1080 (llc1080) native grid used in the original simulation.'
    ds.attrs['id'] = '10.5067/XXXXX-XXXXX' # will update with DOI when avail
    ds.attrs['geospatial_vertical_min'] = np.round(ds.Zu.min().values,1)
    ds.attrs['geospatial_lat_min'] = np.round(ds.YC.min().values,1)
    ds.attrs['metadata_link'] = 'https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName=XXXX_L4_GEOMETRY_LLC1080GRID_V1R1' # will update with DOI when avail
    ds.attrs['product_name'] = var_filename_netcdf
    ds.attrs['time_coverage_end'] = str(ds.time_bnds.values[0][0])[:-10]
    ds.attrs['time_coverage_start'] = str(ds.time_bnds.values[0][1])[:-10]
    ds.attrs['product_version'] = 'Version 1, Release 1'
    ds.attrs['program'] = 'NASA Physical Oceanography'
    ds.attrs['source'] = 'The SASSIE ocean model simulation was produced by downscaling the global ECCO state estimate from 1/3 to 1/12 degree grid cells. The ECCO global solution provided initial and boundary conditions and atmospheric forcing.'
    ds.attrs['references'] = 'TBD'
    ds.attrs['summary'] = 'This dataset provides data variable and geometric parameters for the lat-lon-cap 1080 (llc1080) native model grid from the SASSIE ECCO ocean model Version 1 Release 1 (V1r1) ocean and sea-ice state estimate.'
    ds.attrs['title'] = title
    ds.attrs['uuid'] = str(uuid.uuid1())
    
    ## remove some attributes we don't need
    attributes_to_remove = ['product_time_coverage_start', 'product_time_coverage_end',\
                            'geospatial_lat_resolution', 'geospatial_lon_resolution']
    
    ## add current time and date
    current_time = datetime.now().isoformat()[0:19]
    ds.attrs['date_created'] = current_time
    ds.attrs['date_modified'] = current_time
    ds.attrs['date_metadata_modified'] = current_time
    ds.attrs['date_issued'] = current_time
    
    for attr in attributes_to_remove:
        ds.attrs.pop(attr, None)
        
    return ds


def reorder_dims(xr_dataset):
    ## specify order of dims
    tmp = xr_dataset[["time","j","i","k","j_g","i_g","k_u","k_l","k_p1","nv","nb"]]
    tmp = tmp.drop_indexes(["nv","nb"]).reset_coords(["nv","nb"], drop=True)
    
    ## reassign dataset to new dims
    xr_ds_ordered = tmp.assign(xr_dataset)
    
    return xr_ds_ordered


def save_sassie_netcdf_to_s3(var_HHv2_ds, output_dir, root_filename, var_filename_netcdf, var_name):
    ## save netCDF files
    
    ## create encoding
    encoding_var = create_encoding(var_HHv2_ds, output_array_precision = np.float32)
    
    ## stage netcdf on tmp directory on ec2
    tmp_netcdf_dir = "/home/jpluser/sassie/tmp_netcdf"
    var_HHv2_ds.to_netcdf(tmp_netcdf_dir / var_filename_netcdf, encoding = encoding_var)
    var_HHv2_ds.close()
    
    ## push file to s3 cloud
    mybucket = "ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/" + var_name + "_AVG_DAILY"
    cmd=f"aws s3 cp {tmp_netcdf_dir} s3://{mybucket}/ --recursive --include '*.nc'"
    
    ## remove tmp file
    os.system(f"rm -rf {tmp_netcdf_dir}/*")
    
    print('\n==== saved netcdf: ' + var_filename_netcdf + ' ====\n')


def plot_sassie_HHv2_3D(face_arr, depth_level=0, vmin=None, vmax=None,\
    cmap='jet', axs = None, \
    show_colorbar=True):

    tmp = combine_sassie_N1_faces_to_HHv2_3D(face_arr)

    if vmin == None:
        vmin = np.min(tmp)
    if vmax == None:
        vmax = np.max(tmp)

    if axs == None:
        plt.imshow(tmp[depth_level,:,:], origin='lower', interpolation='none',vmin=vmin,vmax=vmax, cmap=cmap)
        if show_colorbar:
            plt.colorbar()

    else:
        im1 = axs.imshow(tmp[depth_level,:,:], origin='lower', interpolation='none',vmin=vmin,vmax=vmax, cmap=cmap)
        fig = plt.gcf()
        if show_colorbar:
            fig.colorbar(im1, ax=axs)


def create_HH_netcdfs(var, data_dir_ec2, metadata_dict, sassie_n1_geometry_ds, vars_table, root_dest_s3_name):
    
    ## loop through each variable that was requested --------------------------------------------
    print('#### ==== processing:', var, '==== #### \n')
    
    ## get root directory for variable and then define directory
    var_tmp_table = vars_table[vars_table.variable.isin([var])]
    root_filename = var_tmp_table.root_filename.values[0]
    
    ## loop through files in root directory
    data_files = np.sort(list(data_dir_ec2.glob('*.data')))
    
    for file in data_files:
        print('loading file: ', file)
        
        ## get filename
        filename = str(file).split('/')[-1]
        
        ## 3D data processing
        if var_tmp_table['n_dims'].values == '3D':
            ## process dataset
            var_HHv2_ds = process_3D_variable(data_dir_ec2, filename, var_tmp_table,\
                                              vars_table, sassie_n1_geometry_ds)
        ## 2D data processing 
        elif var_tmp_table['n_dims'].values == '2D':
            ## process dataset
            var_HHv2_ds = process_2D_variable(data_dir_ec2, filename, var_tmp_table,\
                                              vars_table, sassie_n1_geometry_ds)
        
        ## mask land cells
        var_HHv2_ds = mask_dry_grid_cells(var_HHv2_ds, var, sassie_n1_geometry_ds, grid_point=var_tmp_table['cgrid_point'].values)
        
        ## add metadata
        global_latlon_metadata = metadata_dict['ECCOv4r4_global_metadata_for_all_datasets'] + metadata_dict['ECCOv4r4_global_metadata_for_latlon_datasets']
        var_HHv2_ds = ecco.add_global_metadata(global_latlon_metadata, var_HHv2_ds, var_tmp_table['n_dims'].values[0])
        var_HHv2_ds = ecco.add_coordinate_metadata(metadata['ECCOv4r4_coordinate_metadata_for_latlon_datasets'], var_HHv2_ds, less_output=True)
        var_HHv2_ds, grouping_keywords = ecco.add_variable_metadata(metadata['ECCOv4r4_geometry_metadata_for_latlon_datasets'], var_HHv2_ds, less_output=True)
        var_HHv2_ds, grouping_keywords = ecco.add_variable_metadata(metadata['ECCOv4r4_variable_metadata'], var_HHv2_ds, less_output=True)
        
        ## generate filename
        center_time = var_HHv2_ds.time.values
        yyyy_mm_dd = str(center_time)[2:6] + "-" + str(center_time)[7:9] + "-" + str(center_time)[10:12]
        var_filename_netcdf = var + "_day_mean_" + yyyy_mm_dd + "_ECCO_SASSIE_V1_HH_llc1080.nc"
        
        ## tweak some of the global attributes
        var_HHv2_ds = modify_metadata(var_HHv2_ds, var, var_filename_netcdf)
        
        ## reorder dims
        var_HHv2_ds_ordered = reorder_dims(var_HHv2_ds)
        
        ## save netcdf
        save_sassie_netcdf_to_s3(var_HHv2_ds_ordered, root_dest_s3_name, root_filename, var_filename_netcdf, var)
        
    # return(var_HHv2_ds_final)
    print("######## processing complete ########")


########### Create final routine to process files ########### 

## Specify root directory and process all variables in that dataset

def generate_sassie_ecco_netcdfs(root_filenames, root_s3_name, root_dest_s3_name, files_to_process):
    
    ## --------------------------------------------
    ## open model geometry from ec2
    sassie_n1_geometry_ds = xr.open_dataset('/home/jpluser/sassie/GRID_GEOMETRY_SASSIE_HH_V1R1_NATIVE_LLC1080.nc')
    
    ## open table that includes metadata for all variables
    vars_table = pd.read_csv('/home/jpluser/git_repos/SASSIE_downscale_ecco_v5/sassie_variables_table.csv', index_col=False)
    
    ## --------------------------------------------
    ## load metadata 
    metadata_json_dir = '/home/jpluser/git_repos/ECCO-ACCESS/metadata/ECCOv4r4_metadata_json/'
    
    metadata_fields = ['ECCOv4r4_global_metadata_for_all_datasets',
                       'ECCOv4r4_global_metadata_for_latlon_datasets',
                       'ECCOv4r4_global_metadata_for_native_datasets',
                       'ECCOv4r4_coordinate_metadata_for_1D_datasets',
                       'ECCOv4r4_coordinate_metadata_for_latlon_datasets',
                       'ECCOv4r4_coordinate_metadata_for_native_datasets',
                       'ECCOv4r4_geometry_metadata_for_latlon_datasets',
                       'ECCOv4r4_geometry_metadata_for_native_datasets',
                       'ECCOv4r4_groupings_for_1D_datasets',
                       'ECCOv4r4_groupings_for_latlon_datasets',
                       'ECCOv4r4_groupings_for_native_datasets',
                       'ECCOv4r4_variable_metadata',
                       'ECCOv4r4_variable_metadata_for_latlon_datasets',
                       'ECCOv4r4_dataset_summary']
    
    ## load metadata
    metadata_dict = dict()
    
    for mf in metadata_fields:
        mf_e = mf + '.json'
        # print(mf_e)
        with open(Path(metadata_json_dir + mf_e), 'r') as fp:
            metadata_dict[mf] = json.load(fp)
    
    ## --------------------------------------------
    ## loop through gz files in root directory and process all variables included in the dataset
    
    ## get list of gz files in s3 directory
    s3 = []
    s3 = s3fs.S3FileSystem(anon=False)
    
    # find filenames
    file_list = np.sort(s3.glob(f'{root_s3_name}{root_filenames}/*tar.gz'))

    # construct url form of filenames
    data_urls = [
            's3://' + f
            for f in file_list
        ]
    
    ## specify start and end indices or process all files   
    if len(files_to_process) == 2: # two numbers indicates a range (two indices)
        data_urls_select = data_urls[files_to_process[0]:files_to_process[1]]
    elif len(files_to_process) == 1 and files_to_process[0] == -1: # process all files
        data_urls_select = data_urls
    elif len(files_to_process) == 1 and files_to_process[0] >= 0: # process one file using number as index
        data_urls_select = data_urls[files_to_process[0]]
    else:
        print("invalid entry for `files_to_process` argument")
    
    for data_url in data_urls_select:
        ## download tar.gz file from s3 cloud to ec2 tmp_dir
        s3 = []
        s3 = s3fs.S3FileSystem(anon=False)
        s3.download(data_url, "/home/jpluser/sassie/tmp_gz/" + data_url.split("/")[-1])
        
        ## decompress tar.gz file into *.data and *.meta files
        data_dir_ec2 = Path('/home/jpluser/sassie/tmp_gz/')
        unpack_tar_gz_files(data_dir_ec2)
         
        ## use table to identify which variables are in the dataset
        vars_in_dataset = vars_table[vars_table.root_filename.isin([root_filenames])].variable.values
    
        ## loop through variables in dataset and generate netcdfs
        for var in vars_in_dataset:
        
            ## generate netcdfs for variable
            create_HH_netcdfs(var, data_dir_ec2, metadata_dict, sassie_n1_geometry_ds, vars_table, root_dest_s3_name)
   
        ## after processing is complete, delete data files on ec2
        print("==== deleting ec2 data files ====\n")
        
        ## remove tmp tar.gz files
        os.system(f"rm -rf {str(data_dir_ec2)}/*")
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--root_filenames", action="store",
                        help="The parent directory and root of each filename for each dataset (e.g. ocean_state_3D_day_mean).",
                        dest="root_filenames", type=str, required=True)

    parser.add_argument("-s", "--root_s3_name", action="store",
                        help="The s3 bucket name where all data files are stored (e.g. s3://ecco-model-granules/SASSIE/N1/).", 
                        dest="root_s3_name", type=str, required=True)

    parser.add_argument("-d", "--root_dest_s3_name", action="store",
                        help="The destination s3 bucket where processed netcdfs will be stored (e.g., s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/).", 
                        dest="root_dest_s3_name", type=str, required=True)

    parser.add_argument("-p", "--files_to_process", action="store",
                        help="String specifying whether to process all files (-1), one file (one number as index), or range (start end).",
                        dest="files_to_process", type=list, required=False, default = [-1])

    args = parser.parse_args()
    root_filenames = args.root_filenames
    root_s3_name = args.root_s3_name
    root_dest_s3_name = args.root_dest_s3_name
    files_to_process = args.files_to_process

    generate_sassie_ecco_netcdfs(root_filenames, root_s3_name, root_dest_s3_name, files_to_process)
