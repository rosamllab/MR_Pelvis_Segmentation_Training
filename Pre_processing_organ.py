# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

import os
from PIL import Image
import numpy as np
from glob import glob
import pydicom
from pydicom.data import get_testdata_file
from PIL import Image
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
from skimage.draw import polygon
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



def load_scan(path):
    """
    Loads all DICOM images in path into a list for manipulation.
    

    Parameters
    ----------
    path : string
        The path to a directory containing DICOM files.

    Returns
    -------
    slices : list
        List of DICOM slices sorted according to their location on the patient axis from inferior to superior.

    """
    
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path) if '.dcm' in s]
    slices.sort(key = lambda x: int(x.SliceLocation))
    
    # Get the slice thickness to use for mapping with RTstruct DICOM file
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    
    return slices




def read_structures(RTstruct_path):
    """
    Reads DICOM RTstruct file with contours and returns a list of dictionaries 
    for each contour with key, value pair for the contour color, number, name,
    and contour data.

    Parameters
    ----------
    structure : DICOM RTstruct dataset
        DICOM RTstruct dataset.

    Returns
    -------
    all_contours : list
        A list of dictionaries.

    """
    try:
        struct_file = [RTstruct_path + s for s in os.listdir(RTstruct_path) if '.dcm' in s]
        struct_file_path = os.path.relpath(struct_file[0])
        structure = pydicom.read_file(struct_file_path, force=True)
    except:
        print("Error reading file\nRTstruct path: ", RTstruct_path)
    all_contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        all_contours.append(contour)
    return all_contours


def get_pixels(slices):
    """
    Extracts pixel data from DICOM slices and returns them as a stacked numpy array.

    Parameters
    ----------
    scans : list
        List of DICOM slices.

    Returns
    -------
    numpy array
        Numpy array of the pixel data of the slices.

    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    
    return np.array(image, dtype = np.int16)



def save_imgs_to_npy(imgs, output_path, patient_number):
    """
    Saves the images as a .npy file.

    Parameters
    ----------
    imgs : Stack of numpy arrays     
    
    output_path : String

    Returns
    -------
    None.

    """
    try:
        os.makedirs(output_path)
    except:
        pass
    
    for i in range(len(imgs)):       
        np.save(output_path + "Images_%d.npy" %i, imgs[i])
        
def save_masks_to_npy(imgs, output_path, patient_number):
    """
    Saves the masks as a .npy file.

    Parameters
    ----------
    imgs : Stack of numpy arrays
    output_path : String

    Returns
    -------
    None.

    """
    try:
        os.makedirs(output_path)
    except:
        pass
    
    for i in range(len(imgs[1,1,:])):
        np.save(output_path + "Masks_%d.npy" %i, imgs[:,:, i])

def create_mask_from_contour(contours, slices, image, ROIName):
    """
    Creates a mask from a contour data.

    Parameters
    ----------
    contours : list
        A list of dictionaries of contour data.
    slices : list
        A list of DICOM slices.
    image : numpy.ndarray
        A stack of numpy arrays of the pixel data for the slices.

    Returns
    -------
    label : numpy.ndarray
        A stack of numpy arrays of masks created for each slice.

    """

    
    z = [round(s.ImagePositionPatient[2], 1) for s in slices]
    zNew = [round(s.ImagePositionPatient[2], 2) for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]
    
    label = np.zeros_like(image, dtype=np.uint8)
    
    ROIAliases = {'femr_rt': ['O_Fmr_Rt', 'O_Femr_Rt'], 'femr_lt' : ['O_Femr_Lt'], 'bladder' : ['O_Bldr', 'O_Bldr_Full'], 'rectum':['O_Rctm', 'O_Rctm_Full', 'O_Rctm on ViewRay PRECISE']}
    

    for con in contours:
        if con['name'] in  ROIAliases[ROIName.lower()]:
            for c in con['contours']:
                nodes = np.array(c).reshape((-1,3))
                try:
                    z_index = z.index(np.around(nodes[0,2], 1))
                except:
                    z_index = zNew.index(np.around(nodes[0,2], 2))
                    
                r = (nodes[:,1] - pos_r) / spacing_r
                c = (nodes[:, 0] - pos_c) / spacing_c
                rr, cc = polygon(r,c)
                label[rr, cc, z_index] = 1

    return label




def preprocess_a_patient(data_path, patient_number, ROIName):
    """
    Generates and saves binary masks for dicom slices in data_path.

    Parameters
    ----------
    data_path : String
        The path where the patient directories are located.
    patient_number : int
        Which patient's data to process.

    Returns
    -------
    None.

    """
    
    slices_path = ""
    RTstruct_path = ""
    patient_path = data_path.format(patient_number = patient_number)
    patient_data_path = glob(patient_path + "\\*\\")
    if len(patient_data_path) == 1:
        path_list = glob(patient_data_path[0] + "\\*\\")
        for path in path_list:
            if 'RTst' in path:
                new_path_name = patient_data_path[0] + "\\Patient.{}RTstruct\\".format(patient_number)
                os.rename(path, new_path_name)
                RTstruct_path = new_path_name
            elif 'MR' in path:
                new_mr_path = patient_data_path[0] + "\\Patient_{patient_number}_AM120{patient_number}_MR\\".format(patient_number = patient_number)
                os.rename(path, new_mr_path)
                slices_path = new_mr_path
    elif len(patient_data_path) > 1:
        for the_path in patient_data_path:
            path_list = glob(the_path + "\\*\\")
            for path in path_list:
                if 'RTst' in path:
                    new_path_name = patient_data_path[0] + "\\Patient.{}RTstruct\\".format(patient_number)
                    os.rename(path, new_path_name)                    
                    RTstruct_path = new_path_name
                elif 'MR' in path:
                    new_mr_path = patient_data_path[0] + "\\Patient_{patient_number}_AM120{patient_number}_MR\\".format(patient_number = patient_number)
                    os.rename(path, new_mr_path)
                    slices_path = new_mr_path
    
    
    pixels_path = os.path.join(patient_path, "Extracted\\Pixel_arrays\\")
    mask_output_path = os.path.join(patient_path, "Extracted\\Masks_{ROIName}\\").format(ROIName = ROIName)
        

    if not os.path.exists(pixels_path):
        os.makedirs(pixels_path)
    else:
        pass
    
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)
    else:
        pass
    try:
        slices = load_scan(slices_path)
        pixel_arrays = get_pixels(slices)
    except:
        print("Patient_number: ", patient_number)
    save_imgs_to_npy(pixel_arrays, pixels_path, patient_number)
    try:
        contours = read_structures(RTstruct_path)
    except:
        print("Patient_number: ", patient_number)
        print("RTstruct_path: ", RTstruct_path)    
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    label = create_mask_from_contour(contours, slices, image, ROIName)
    save_masks_to_npy(label, mask_output_path, patient_number)

        
    
    
def preprocess_all_patients(start, end, ROIName):
    parent_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(parent_path, "Data\\Test_Data\\Patient_{patient_number}")
    #exclude = {'rectum':[11,13,16,24,32,35,37], 'bladder':[], 'femr_rt':[], 'femr_lt': []}
    
    for i in tqdm(range(start, end+1)):
        preprocess_a_patient(data_path, i, ROIName)
        # if i not in exclude[ROIName]:
        #     preprocess_a_patient(data_path, i, ROIName)


if __name__ == "__main__":
    preprocess_all_patients(1,75, "bladder")





    
