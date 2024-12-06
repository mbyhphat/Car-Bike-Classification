from skimage.feature import hog
import os
import cv2
import matplotlib.pyplot as plt 

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features
    
category_map = {'Bike': 0, 'Car': 1}
root_dir = 'data/Car-Bike-Dataset'
hog_features = []

for category, label in category_map.items():
    category_path = os.path.join(root_dir,category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path,img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feature = get_hog_features(img)
        
