# Copyright 2024 Diagnostic Image Analysis Group, Radboud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import SimpleITK as sitk
import time
import os
import re
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import binary_dilation

def parse_radiomics_features(feature_names):
    """
    Parse an array of radiomic feature names and organize them into a list of individual feature dictionaries.
    
    Args:
        feature_names (list): List of radiomic feature names
        
    Returns:
        list: List of dictionaries, each representing a single feature with its properties
              Structure: [
                 {
                     'tag': 'tag_value',
                     'image_type': 'original',
                     'feature_class': 'firstorder',
                     'feature_name': 'Mean',
                     'parameters': {}
                 },
                 {
                     'tag': 'tag_value',
                     'image_type': 'wavelet',
                     'feature_class': 'glcm', 
                     'feature_name': 'Correlation',
                     'parameters': {'direction': 'HHL'}
                 },
                 ...
              ]
    """
    features_list = []
    
    # Track available directions and sigmas for summary
    wavelet_directions = set()
    log_sigmas = set()
    
    for feature_name in feature_names:
        feature_dict = {}
        
        # First, separate the tag from the rest
        if '_' not in feature_name:
            continue
            
        parts = feature_name.split('_', 1)
        feature_dict['tag'] = parts[0]
        remaining = parts[1]
        
        # Separate the image type from feature class and name
        if '_' not in remaining:
            continue
            
        # Handle the image type part (before first underscore)
        parts = remaining.split('_', 1)
        image_type_part = parts[0]
        feature_part = parts[1]
        
        # Further split the feature part into class and name
        if '_' not in feature_part:
            continue
            
        feature_parts = feature_part.split('_', 1)
        feature_dict['feature_class'] = feature_parts[0]
        feature_dict['feature_name'] = feature_parts[1]
        
        # Determine the base image type and extract parameters
        if image_type_part == 'original':
            feature_dict['image_type'] = 'original'
            feature_dict['parameters'] = {}
        elif image_type_part.startswith('wavelet-'):
            feature_dict['image_type'] = 'wavelet'
            
            # Extract wavelet direction (e.g., 'HHH', 'HHL', etc.)
            direction_match = re.search(r'wavelet-([HL]+)', image_type_part)
            if direction_match:
                direction = direction_match.group(1)
                wavelet_directions.add(direction)
                feature_dict['parameters'] = {'direction': direction}
            else:
                feature_dict['parameters'] = {}
        elif image_type_part.startswith('log-sigma-'):
            feature_dict['image_type'] = 'log'
            
            # Extract sigma value
            sigma_match = re.search(r'log-sigma-([0-9\.\-]+)', image_type_part)
            if sigma_match:
                # Convert format like '1-0' to '1.0'
                sigma_str = sigma_match.group(1).replace('-', '.')
                # Remove any trailing periods that might cause float conversion errors
                sigma_str = sigma_str.rstrip('.')
                sigma = float(sigma_str)
                log_sigmas.add(sigma)
                feature_dict['parameters'] = {'sigma': sigma}
            else:
                feature_dict['parameters'] = {}
        elif image_type_part.startswith('lbp-'):
            # Handle LBP features
            if 'lbp-2d' in image_type_part.lower():
                feature_dict['image_type'] = 'lbp-2d'
                feature_dict['parameters'] = {'force2D': True}
            elif 'lbp-3d' in image_type_part.lower():
                feature_dict['image_type'] = 'lbp-3d'
                feature_dict['parameters'] = {}
            else:
                feature_dict['image_type'] = 'lbp'
                feature_dict['parameters'] = {}
        else:
            # Other image types (square, squareroot, etc.)
            feature_dict['image_type'] = image_type_part
            feature_dict['parameters'] = {}
        
        # Store the original feature name for reference
        feature_dict['original_name'] = feature_name
        
        # Add to the list of features
        features_list.append(feature_dict)
    
    # Create a summary dictionary for easy access to all directions and sigmas
    summary = {
        'wavelet_directions': sorted(list(wavelet_directions)),
        'log_sigmas': sorted(list(log_sigmas))
    }
    
    return features_list, summary

def generate_radiomics_config_from_features(features_list, summary):
    """
    Generate a configuration dictionary for PyRadiomics based on parsed features.
    
    Args:
        features_list (list): List of feature dictionaries as returned by parse_radiomics_features
        summary (dict): Summary dictionary with collected directions and sigmas
        
    Returns:
        dict: Configuration dictionary with imageType, featureClass, and settings
    """
    config = {
        'imageType': {},
        'featureClass': {},
        'setting': {}
    }
    
    # Keep track of image types and feature classes
    image_types = {}
    feature_classes = {}
    
    # Process all features to collect unique image types and feature classes
    for feature in features_list:
        image_type = feature['image_type']
        feature_class = feature['feature_class']
        feature_name = feature['feature_name']
        parameters = feature['parameters']
        
        # Handle image types
        if image_type not in image_types:
            if image_type == 'original':
                image_types[image_type] = {}
            elif image_type == 'wavelet' and 'wavelet_directions' in summary:
                image_types[image_type] = {
                    'waveletDirection': summary['wavelet_directions']
                }
            elif image_type == 'log' and 'log_sigmas' in summary:
                image_types[image_type] = {
                    'sigma': summary['log_sigmas']
                }
            elif image_type == 'lbp-2d':
                image_types[image_type] = {}
                # Set force2D in settings
                if 'setting' not in config:
                    config['setting'] = {}
                config['setting']['force2D'] = True
            elif image_type == 'lbp-3d':
                image_types[image_type] = {}
            else:
                image_types[image_type] = {}
        
        # Handle feature classes
        if feature_class not in feature_classes:
            feature_classes[feature_class] = set()
        
        feature_classes[feature_class].add(feature_name)
    
    # Convert to PyRadiomics format
    for image_type, params in image_types.items():
        if image_type == 'original':
            config['imageType']['Original'] = params
        elif image_type == 'wavelet':
            config['imageType']['Wavelet'] = params
        elif image_type == 'log':
            config['imageType']['LoG'] = params
        elif image_type == 'lbp-2d':
            config['imageType']['LBP2D'] = params
        elif image_type == 'lbp-3d':
            config['imageType']['LBP3D'] = params
        else:
            # Handle other image types with first letter capitalized
            config['imageType'][image_type.capitalize()] = params
    
    # Add feature classes to config
    for feature_class, features in feature_classes.items():
        config['featureClass'][feature_class] = list(features)
    
    return config


def whole_panc_lbl(itkimg):
    """
    Extract whole pancreas segmentation and eliminate extra labels.

    Parameters:
    - itkimg (SimpleITK.Image): Segmentation label as ITK Image object.
    
    Returns:
    - outimg (SimpleITK.Image): Processed label.
    """
    imgarr = sitk.GetArrayFromImage(itkimg)
    imgarr[imgarr==1] = 4
    imgarr[imgarr==5] = 4
    imgarr[imgarr!=4] = 0
    imgarr[imgarr==4] = 1
    outimg = sitk.GetImageFromArray(imgarr)
    outimg.CopyInformation(itkimg)
    return outimg

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, out_size=[], out_origin=[], out_direction=[], mask=None):
    """
    Resample an image to a new spacing, optionally using a mask to restrict interpolation to foreground regions.
    
    Args:
        itk_image: SimpleITK image to resample
        out_spacing: Output spacing [x,y,z]
        is_label: Whether the image is a label map (use nearest neighbor)
        out_size: Output size (calculated from spacing if not provided)
        out_origin: Output origin (uses input image origin if not provided)
        out_direction: Output direction (uses input image direction if not provided)
        mask: Optional mask to restrict interpolation (interpolation only within mask, background preserved)
        
    Returns:
        Resampled SimpleITK image
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    if not out_size:
        out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                   int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                   int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    # Set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    
    if not out_direction:
        out_direction = itk_image.GetDirection()
    resample.SetOutputDirection(out_direction)
    
    if not out_origin:
        out_origin = itk_image.GetOrigin()
    resample.SetOutputOrigin(out_origin)
    
    resample.SetTransform(sitk.Transform())
    minvalue = int(np.nanmin(sitk.GetArrayFromImage(itk_image)))
    resample.SetDefaultPixelValue(minvalue)
    
    # If we're resampling a label or we don't have a mask, use the standard approach
    if is_label or mask is None:
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
        return resample.Execute(itk_image)
    
    # If we have a mask, we'll do masked resampling
    # First, resample the image using the desired interpolator
    resample.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resample.Execute(itk_image)
    
    # Next, resample the mask using nearest neighbor to keep binary values
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = resample.Execute(mask)
    
    # Now, resample the image using nearest neighbor for comparison
    # This preserves original values without interpolation for background
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    nn_resampled = resample.Execute(itk_image)
    
    # Convert to arrays for manipulation
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    nn_resampled_array = sitk.GetArrayFromImage(nn_resampled)
    mask_array = sitk.GetArrayFromImage(resampled_mask)
    
    # Create the final image: use interpolated values where mask=1, nearest neighbor elsewhere
    final_array = np.where(mask_array > 0, resampled_array, nn_resampled_array)
    
    # Convert back to SimpleITK image
    result = sitk.GetImageFromArray(final_array)
    result.CopyInformation(resampled_image)
    
    return result


def CropPancreasROI(image, low_res_segmentation, margins):
     
    pancreas_mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    assert(len(np.unique(pancreas_mask_np))==2)    
    
    pancreas_mask_nonzeros = np.nonzero(pancreas_mask_np)
    
    min_x = min(pancreas_mask_nonzeros[2])
    min_y = min(pancreas_mask_nonzeros[1])
    min_z = min(pancreas_mask_nonzeros[0])
    
    max_x = max(pancreas_mask_nonzeros[2])
    max_y = max(pancreas_mask_nonzeros[1])
    max_z = max(pancreas_mask_nonzeros[0])
    
    start_point_coordinates = (int(min_x), int(min_y), int(min_z))
    finish_point_coordinates = (int(max_x), int(max_y), int(max_z))          
    
    start_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(start_point_coordinates)
    finish_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(finish_point_coordinates)
    
    start_point = image.TransformPhysicalPointToIndex(start_point_physical)
    finish_point = image.TransformPhysicalPointToIndex(finish_point_physical)


    spacing = image.GetSpacing()
    size = image.GetSize()
        
    marginx = int(margins[0]/spacing[0])
    marginy = int(margins[1]/spacing[1])
    marginz = int(margins[2]/spacing[2])
    
    x_start = max(0, start_point[0] - marginx)
    x_finish = min(size[0], finish_point[0] + marginx)
    y_start = max(0, start_point[1] - marginy)
    y_finish = min(size[1], finish_point[1] + marginy)
    z_start = max(0, start_point[2] - marginz)
    z_finish = min(size[2], finish_point[2] + marginz)
    
    cropped_image = image[x_start:x_finish, y_start:y_finish, z_start:z_finish]
    cropped_label = low_res_segmentation[min_x:max_x,min_y:max_y,min_z:max_z]
    crop_coordinates = {'x_start': x_start,
                        'x_finish': x_finish,
                        'y_start': y_start,
                        'y_finish': y_finish,
                        'z_start': z_start,
                        'z_finish': z_finish}
      
    return cropped_image, crop_coordinates,cropped_label

def PostProcessing(cropped_prediction, pred_path_nifti):
    pancreas_mask = sitk.ReadImage(pred_path_nifti)
    pancreas_mask_np = sitk.GetArrayFromImage(pancreas_mask)
    pancreas_mask_np[pancreas_mask_np==1] = 4
    pancreas_mask_np[pancreas_mask_np==5] = 4
    pancreas_mask_np[pancreas_mask_np!=4] = 0
    pancreas_mask_np[pancreas_mask_np==4] = 1

    kernel = np.ones((5, 5, 5), dtype=bool)
    dilated_pancreas_mask = binary_dilation(pancreas_mask_np, structure=kernel)

    prediction_np = cropped_prediction['probabilities'][1]
    prediction_np = prediction_np.astype(np.float32)

    prediction_np[dilated_pancreas_mask!=1] = 0

    return prediction_np

def GetFullSizDetectionMap(prediction_np, cropp_coordinates, full_image,baseline_mode=False):
    if baseline_mode:
        lesion_candidates, confidences, indexed_pred = extract_lesion_candidates(prediction_np)
    else:
        lesion_candidates, confidences, indexed_pred = extract_lesion_candidates(prediction_np,threshold='dynamic-v2',dynamic_threshold_factor=15)


    patient_level_prediction = float(np.max(lesion_candidates))


    full_size_detection_map = np.zeros(sitk.GetArrayFromImage(full_image).shape)
    full_size_detection_map = full_size_detection_map.astype(np.float32)


    # Use integer slicing, ensuring no slice is empty
    z_slice = slice(int(cropp_coordinates['z_start']), int(cropp_coordinates['z_finish']))
    y_slice = slice(int(cropp_coordinates['y_start']), int(cropp_coordinates['y_finish']))
    x_slice = slice(int(cropp_coordinates['x_start']), int(cropp_coordinates['x_finish']))

    full_size_detection_map[z_slice, y_slice, x_slice] = lesion_candidates
    full_size_detection_map = full_size_detection_map.astype(np.float32)

    detection_map_image = sitk.GetImageFromArray(full_size_detection_map)
    detection_map_image.CopyInformation(full_image)

    return detection_map_image, patient_level_prediction

def GetFullSizDetectionMap_nifti(prediction_np, cropp_coordinates, full_image):
    lesion_candidates = prediction_np
    full_size_detection_map = np.zeros(sitk.GetArrayFromImage(full_image).shape)
    full_size_detection_map = full_size_detection_map.astype(np.float32)


    # Use integer slicing, ensuring no slice is empty
    z_slice = slice(int(cropp_coordinates['z_start']), int(cropp_coordinates['z_finish']))
    y_slice = slice(int(cropp_coordinates['y_start']), int(cropp_coordinates['y_finish']))
    x_slice = slice(int(cropp_coordinates['x_start']), int(cropp_coordinates['x_finish']))

    full_size_detection_map[z_slice, y_slice, x_slice] = lesion_candidates
    full_size_detection_map = full_size_detection_map.astype(np.float32)

    detection_map_image = sitk.GetImageFromArray(full_size_detection_map)
    detection_map_image.CopyInformation(full_image)

    return detection_map_image
        
def GetIndexRangeInROI(roi_mask, target_value=1):
    x, y, z = np.where(roi_mask ==1)
    x = np.unique(x)
    y = np.unique(y)
    z = np.unique(z)
    return x, y, z

def GetIndexRangeInROI2(roi_mask, target_value=1):
    x, y, z = np.where(roi_mask != 0)
    x = np.unique(x)
    y = np.unique(y)
    z = np.unique(z)
    return x, y, z
    
def GenerateFeatureROI(roi_mask, feature_map):
    # print(sorted(self.original_feature_map_array.flatten(), reverse=True))
    x, y, z = GetIndexRangeInROI(roi_mask)
    xf,yf,zf = GetIndexRangeInROI2(feature_map)
    rangex = max(xf)-min(xf)+1
    rangey = max(yf)-min(yf)+1
    rangez = max(zf)-min(zf)+1
    feature_roi = np.zeros_like(roi_mask)
    feature_roi = feature_roi.astype(np.float32, copy=False)
    feature_roi[np.min(x):np.min(x) + rangex,
                np.min(y):np.min(y) + rangey,
                np.min(z):np.min(z) + rangez]= feature_map[min(xf):max(xf)+1,
                                                           min(yf):max(yf)+1,
                                                           min(zf):max(zf)+1]
    return feature_roi

def create_feature_specific_extractors(base_extractor, feature_names):
    """
    Creates a list of feature extractors, each configured to extract just one specific feature.
    
    Args:
        base_extractor (radiomics.featureextractor.RadiomicsFeatureExtractor): Base extractor to copy settings from
        feature_names (list): List of feature names to create extractors for
        
    Returns:
        list: List of tuples, each containing (feature_name, specialized_extractor)
    """
    # Parse feature names
    features_list, summary = parse_radiomics_features(feature_names)
    
    # List to store the specialized extractors
    specialized_extractors = []
    
    # Create a specialized extractor for each feature
    for feature in features_list:
        # Create configuration for this single feature
        config = {
            'imageType': {},
            'featureClass': {}
        }
        
        # Configure image type based on feature's image type
        image_type = feature['image_type']
        
        if image_type == 'original':
            config['imageType']['Original'] = {}
        elif image_type == 'wavelet':
            # For wavelet, only include the specific direction
            if 'direction' in feature['parameters']:
                direction = feature['parameters']['direction']
                config['imageType']['Wavelet'] = {
                    'waveletDirection': [direction]
                }
            else:
                config['imageType']['Wavelet'] = {}
        elif image_type == 'log':
            # For LoG, only include the specific sigma
            if 'sigma' in feature['parameters']:
                sigma = feature['parameters']['sigma']
                config['imageType']['LoG'] = {
                    'sigma': [sigma]
                }
            else:
                config['imageType']['LoG'] = {}
        elif image_type == 'lbp-2d':
            # For LBP-2D, set force2D to True
            config['imageType']['LBP2D'] = {}
            # We'll set force2D later after the extractor is created
        elif image_type == 'lbp-3d':
            config['imageType']['LBP3D'] = {}
        else:
            # Other image types
            config['imageType'][image_type.capitalize()] = {}
        
        # Configure feature class to only include this feature
        feature_class = feature['feature_class']
        feature_name = feature['feature_name']
        config['featureClass'] = {
            feature_class: [feature_name]
        }
        
        # Create a new extractor with the base settings
        specialized_extractor = type(base_extractor)()
        
        # Copy settings from the base extractor
        for setting, value in base_extractor.settings.items():
            specialized_extractor.settings[setting] = value
        
        # Apply special settings based on image type
        if image_type == 'lbp-2d':
            specialized_extractor.settings['force2D'] = True
            
        # Apply the specific configuration
        specialized_extractor.enabledImagetypes = {}
        specialized_extractor.enabledFeatures = {}
        
        # Enable the specific image type
        for image_type_key, params in config['imageType'].items():
            specialized_extractor.enableImageTypeByName(image_type_key, enabled=True, customArgs=params)
        
        # Enable the specific feature
        for feature_class_key, features in config['featureClass'].items():
            if features:  # If specific features are specified
                specialized_extractor.enabledFeatures[feature_class_key] = features
            else:  # Enable all features in this class
                specialized_extractor.enableFeatureClassByName(feature_class_key, enabled=True)
        
        # Store the specialized extractor with its corresponding feature name
        specialized_extractors.append((feature['original_name'], specialized_extractor))
    
    return specialized_extractors





