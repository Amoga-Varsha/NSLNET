import numpy as np
import time 
from skimage import io, color, feature, util

#Using Grey level co-occurence matrix to find the features from the image.

st = time.time()

image = io.imread('image2test.jpg') # Load image
gs = util.img_as_ubyte(color.rgb2gray(image)) # Convert image to grayscale and uint8 format
distances = [1, 2, 3] # Define distance and angle offsets for GLCM
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

glcm = feature.greycomatrix(gs, distances=distances, angles=angles, levels=256, symmetric=True, normed=True) # Calculate GLCM

# Calculate GLCM properties
contrast = feature.greycoprops(glcm, 'contrast')
dissimilarity = feature.greycoprops(glcm, 'dissimilarity')
homogeneity = feature.greycoprops(glcm, 'homogeneity')
energy = feature.greycoprops(glcm, 'energy')
correlation = feature.greycoprops(glcm, 'correlation')

# Print GLCM properties
print('GLCM Contrast:', contrast)
print('GLCM Dissimilarity:', dissimilarity)
print('GLCM Homogeneity:', homogeneity)
print('GLCM Energy:', energy)
print('GLCM Correlation:', correlation)

print(time.time() - st)