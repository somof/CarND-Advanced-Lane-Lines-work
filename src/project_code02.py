
# importing some useful packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import cv2

import glob
import scipy as sc
from scipy import stats
from sklearn import linear_model, datasets
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Read in the saved objpoints and imgpoints for Caliblation

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
import pickle
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]



# Helper Functions

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    RGB = cv2.split(img)
    return cv2.addWeighted(RGB[0], 0.5, RGB[1], 0.5, 0.0)

    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, angle_min, angle_max):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    points = []
    filtered_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = math.hypot(x2 - x1, y2 - y1)
                theta = math.atan2((y2 - y1), (x2 - x1))
                if 5 < length and angle_min < theta and theta < angle_max:
                    filtered_lines.append([[x1, y1, x2, y2]])
                    points.extend(devide_points(x1, y1, x2, y2))

                # if 5 < length and  0.05*math.pi < theta and theta <  0.45*math.pi:
                #     filtered_lines.append([[x1, y1, x2, y2]])
                #     right_points.extend(devide_points(x1, y1, x2, y2))

                # if 5 < length and -0.45*math.pi < theta and theta < -0.05*math.pi:
                #     filtered_lines.append([[x1, y1, x2, y2]])
                #     left_points.extend(devide_points(x1, y1, x2, y2))

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, filtered_lines)

    # draw angle scale
    # radial = 200
    # for unit in (0.1, 0.2, 0.3, 0.4):
    #     rad = math.pi * unit
    #     cx = 500
    #     cy = 300
    #     x = int(math.sin(-rad) * radial + cx)
    #     y = int(math.cos(-rad) * radial + cy)
    #     cv2.line(line_img, (cx, cy), (x, y), [255, 0, 255], 1)
    #     cx = 550
    #     x = int(math.sin(rad) * radial + cx)
    #     y = int(math.cos(rad) * radial + cy)
    #     cv2.line(line_img, (cx, cy), (x, y), [0, 255, 255], 1)

    return line_img, points

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1.0, λ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def devide_points(x1, y1, x2, y2):
    length = math.hypot(x2 - x1, y2 - y1)
    num = int(math.sqrt(length))
    lines = []
    for i in range(0, num):
        lines.append([x1 + i * (x2 - x1) / num,
                      y1 + i * (y2 - y1) / num])
    return lines


def regression_line(points, hrange, color, thickness):

    # transform point list
    x = [d[0] for d in points]
    y = [d[1] for d in points]

    # Linear regressor
    # slope, intercept, r_value, _, _ = stats.linregress(x, y)
    # # func = lambda x: x * slope + intercept
    # func = lambda y: int((y - intercept) / slope)
    # cv2.line(img, (func(500), 500), (func(300), 300), color, thickness)

    if 30 < len(x):
        # RANSAC regressor
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                                    min_samples=24,
                                                    residual_threshold=25,
                                                    # is_data_valid=is_data_valid,
                                                    # random_state=0
                                                    )
        if model_ransac is not None:
            X = np.array(x)
            model_ransac.fit(X[:, np.newaxis], y)
            line_x = np.arange(hrange[0], hrange[1], hrange[1] - hrange[0] - 1)
            line_y = model_ransac.predict(line_x[:, np.newaxis])
            return (int(line_x[0]), int(line_y[0])), (int(line_x[1]), int(line_y[1]))

    return [0, 0], [0, 0]


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, sobel_thresh=(0, 255)):
    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the absolute value of the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # apply a threshold, and create a binary image result
    binary_output = np.zeros(absgraddir.shape).astype(np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def process_image(image, weight=0.5):

    # 1) Undistort using mtx and dist
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # 3) Create binary image via Combining Threshold

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, sobel_thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, sobel_thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255

    sximg = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    return sximg

 # cv2.error:
 # /Users/jenkins/miniconda/1/x64/conda-bld/conda_1486587097465/work/opencv-3.1.0/modules/imgproc/src/color.cpp:7935:
 # error: (-215) depth == CV_8U || depth == CV_16U || depth == CV_32F in
 # function cvtColor


    # 3) Sobel filter
    sobel_kernel = 3
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude

    # gradmag = np.sqrt(sobelx**2 + sobely**2)
    # scale_factor = np.max(gradmag) / 255
    # gradmag = (gradmag/scale_factor).astype(np.uint8)  # Rescale to 8 bit
    # Create a binary image of ones where threshold is met, zeros otherwise
    # mag_thresh = [20, 100]
    # binary_output = np.zeros_like(gradmag)
    # binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    dir_thresh = [0.7, 1.3]
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 255



    # Thresholding
    # Color/gradient threshold


    # Perspective Transform
    

    # Detect lane lines
    # -> https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/c41a4b6b-9e57-44e6-9df9-7e4e74a1a49a



    # Determine the lane curvature
    # filter curvature values


    # detect offset of the car position






    # Tracking -> tips

    #


    # Sanity Check
    # - Checking that they have similar curvature
    # - Checking that they are separated by approximately the right distance horizontally
    # - Checking that they are roughly parallel




    # Look-Ahead Filter




    # Reset
    # If your sanity checks reveal that the lane lines you've detected are
    # problematic for some reason, you can simply assume it was a bad or
    # difficult frame of video, retain the previous positions from the frame
    # prior and step to the next frame to search again. If you lose the lines
    # for several frames in a row, you should probably start searching from
    # scratch using a histogram and sliding window, or another method, to
    # re-establish your measurement.


    # Smoothing



    # Drawing -> tips





    sximg = cv2.cvtColor(binary_output, cv2.COLOR_GRAY2RGB)
    return sximg


    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    sximg = cv2.cvtColor(sxbinary, cv2.COLOR_GRAY2RGB)
    return sximg


    # 4) If corners found: 
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)

        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        img_size = (gray.shape[1], gray.shape[0])
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
                
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        offset = 100
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                         [img_size[0]-offset, img_size[1]-offset], 
                         [offset, img_size[1]-offset]])

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size)
        
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    return warped, M





    result = undist
    return result



# process frame by frame

global frame_time
frame_time = 0

white_output = '/Users/ichikihiroshi/CarND-Advanced-Lane-Lines/challenge_video_out.mp4'
cap = cv2.VideoCapture('../challenge_video.mp4')

while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    print('{:5.0f} msec'.format(msec))
    frame_time += 100

    ret, frame = cap.read()
    if ret:
        result = process_image(frame, weight=0.5)
        img = cv2.vconcat([cv2.resize(frame, (800, 380)),
                           cv2.resize(result, (800, 380))])
        cv2.imshow('frame', img)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
exit(0)



# white_output = '/Users/ichikihiroshi/CarND-Advanced-Lane-Lines/project_video_out.mp4'
# clip1 = VideoFileClip('../project_video.mp4')
# white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
# exit(0)


white_output = '/Users/ichikihiroshi/CarND-Advanced-Lane-Lines/challenge_video_out.mp4'
clip1 = VideoFileClip('../challenge_video.mp4')
frames = int(clip1.fps * clip1.duration)
print('frame num: ', frames)
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
exit(0)

white_output = '/Users/ichikihiroshi/CarND-Advanced-Lane-Lines/harder_challenge_video_out.mp4'
clip1 = VideoFileClip('../harder_challenge_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
