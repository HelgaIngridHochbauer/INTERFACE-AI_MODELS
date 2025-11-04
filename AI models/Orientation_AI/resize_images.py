# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 10:30:09 2025

@author: USER
"""

import cv2
import os

"""
Resizes a file given either a maxixum dimension or a scale factor.
If the scale factor is a non zero positive it applies it instead of the maximum dimension.
If the maximum dimension is used an aspect ratio is applied to the image.
Stores the file in the output folder.

Throws a cv2.error in case of errors
"""  
def resize(filename, output_folder, max_dim=20, scale_factor=0):
    try:
    
        image = cv2.imread(filename)
        
        if scale_factor > 0:
            resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)    
        else:
            original_height, original_width = image.shape[:2]
            
            if original_height > original_width: #image is portrait
                aspect_ratio = max_dim / original_height  
                new_width = int(original_width * aspect_ratio)
                new_height = max_dim
            else: #image is landscape
                aspect_ratio = max_dim / original_width
                new_height = int(original_height * aspect_ratio)
                new_width = max_dim
            
            resized_image = cv2.resize(image, (new_width, new_height))
        
        cv2.imwrite(os.path.join(output_folder, filename), resized_image)

    except cv2.error as e:
        print("Error: ", e)
        
    
max_dim = 100
valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

folder = 'C:\\Users\\USER\\Downloads\\helga-cropped-pendants\\'
output_folder = 'C:\\Users\\USER\\Downloads\\helga-cropped-pendants_resized\\'

try:
    os.chdir(folder)
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)) and any(f.lower().endswith(ext) for ext in valid_extensions):
            resize(f, output_folder, scale_factor=0.01)
except OSError as e:
    print("Error: ", e)    
