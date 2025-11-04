import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_data_and_augment(folder):
    row_count = 0
    df = pd.read_csv(folder + 'angles.txt',sep = ' ', header=None)
    row_count = df.shape[0]
    labels = pd.DataFrame(columns = ['name', 'angle'])
    for index, row in df.iterrows():
      print(f"Processing {row[0]}")
      i = int(row[0])
      angle = row[3]
      image = Image.open(folder + f"{i:0>4}" + '.png').convert('RGB')
      for j in range(0,330,30):
        rotated = image.rotate(-j, expand=True)
        name = f"{i:0>4}_rot_{j}"
        rotated.save(folder + "augmented/" + name + '.png')
        labels.loc[len(labels)] = {'name':name, 'angle' : (convert_angle_to_360(angle) + j) % 360}
 
    labels.to_csv(folder + 'augmented/angles.txt', sep = ' ', index = False, header = True)
    
    
def convert_angle_to_360(angle): # our angles are [-180,180] with 0 in East and negative values in South
    angle = 90 - angle
    return angle if angle >= 0 else 360 + angle

if (__name__ == '__main__'):
    get_data_and_augment('C:/Users/USER/datasets/rotation-angles/')
    