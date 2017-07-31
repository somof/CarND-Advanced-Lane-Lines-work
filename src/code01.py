
import os as os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 9  # enter the number of inside corners in x
ny = 6  # enter the number of inside corners in y

# Make a list of calibration images
fname = 'calibration_test.png'
fname = '../camera_cal/calibration3.jpg'
print(os.access(fname, os.R_OK))

img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
print(gray.shape)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
print(ret, corners)



# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()
