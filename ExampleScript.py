# Salami exercise
# Anders Nymark Christensen
# 20230221
# Adapted from work by Anders Bjorholm Dahl

# Folder where your data files are placed
dirIn = 'C:/Users/anym/Downloads/data/'


import helpFunctions as hf 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread
    

## Example of loading a multi spectral image


multiIm, annotationIm = hf.loadMulti('multispectral_day20.mat' , 'annotation_day20.png', dirIn)

# multiIm is a multi spectral image - the dimensions can be seen by
multiIm.shape

## Show image óf spectral band 7

plt.imshow(multiIm[:,:,6])
plt.show()

## annotationIm is a binary image with 3 layers

annotationIm.shape

## In each layer we have a binary image:
# 0 - background with salami
# 1 fat annotation
# 2 meat annotation

# Here we show the meat annotation

plt.imshow(annotationIm[:,:,2])
plt.show()


## The function getPix extracts the multi spectral pixels from the annotation

# Here is an example with meat- and fat annotation
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);

# Here we plot the mean values for pixels with meat and fat respectively
plt.plot(np.mean(meatPix,0),'b')
plt.plot(np.mean(fatPix,0),'r')
plt.show()




## The function showHistogram makes a histogram that is returned as a vector

# Here is an example - last argument tells the function to plot the histogram for meat and fat
h = hf.showHistograms(multiIm, annotationIm[:,:,2:3], 2, 1)


## The histogram is also in h
# But not truncated like in the plot. If we wnat to avoid plotting all 256 dimensions, 
# we can do like below, and only plot the first 50 values

plt.plot(h[0:50:,:])
plt.show()

## The function setImagePix produces a colour image
# where the pixel coordinates are given as input

# Load RGB image
imRGB = imread(dirIn + 'color_day20.png')

# Pixel coordinates for the fat annotation
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1])

# Concatenate the pixel coordinates to a matrix
pixId = np.stack((fatR, fatC), axis=1)

# Make the new images
rgbOut = hf.setImagePix(imRGB, pixId)
plt.imshow(rgbOut)
plt.show()



