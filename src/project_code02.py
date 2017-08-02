
# importing some useful packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import cv2
from PIL import Image

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

def weighted_img(img, initial_img, alpha=0.8, beta=1.0, gamma=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


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


def hls_select(image, hthresh=(0, 255), sthresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(((h_channel >= hthresh[0]) & (h_channel <= hthresh[1])) &
                   ((s_channel >= sthresh[0]) & (s_channel <= sthresh[1])))] = 1
    return binary_output


def warper(img, src, dst):
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped



# source and destination points
# Given src and dst points, calculate the perspective transform matrix

(width, height) = (1280, 720)
center = width / 2 - 10
magx = 320
magy = 0
src = np.float32([[631, 425], [649, 425], [1055, 675], [265, 675]])
src = np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]])
dst = np.float32([[magx, magy], [width - magx, magy], [width-magx, height - magy], [magx, height - magy]])

# src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
# dst = np.float32([[320, 0],   [320, 720], [ 960, 720], [960,   0]])

# src = np.float32([[631, 425], [649, 425], [1055, 675], [265, 675]])
# src = np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]])
# dst = np.float32([[320,   0], [960,   0], [ 960, 720], [320, 720]])

M = cv2.getPerspectiveTransform(src, dst)

# for file in ('straight_lines1.jpg', 'straight_lines2.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg'):
#     print(file)
#     img = cv2.imread('../test_images/' + file)
#     img = cv2.undistort(img, mtx, dist, None, mtx)
#     warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
#     cv2.imshow('frame', warped)
#     # cv2.imwrite('../output_images/be_' + file, warped)
#     if cv2.waitKey(1000) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# exit(0)


# window settings
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 70  # How much to slide left and right for searching


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level*height),
           max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2+int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]
                                       - (level + 1) * window_height):int(image.shape[0]
                                                                          - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset - margin, 0))
        l_max_index = int(min(l_center+offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset - margin, 0))
        r_max_index = int(min(r_center+offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def process_image(image, weight=0.5):

    # 1) Undistort using mtx and dist
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # 3) Create binary image via Combining Threshold

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, sobel_thresh=(20, 80))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, sobel_thresh=(20, 80))
    mag_binary = mag_thresh(gray, sobel_kernel=5, mag_thresh=(20, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))  # (0.7, 1.3))
    hls_binary = hls_select(undist, hthresh=(15, 100), sthresh=(170, 255))

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((dir_binary.shape), dtype=np.uint8)
    # combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 255
    combined[((((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | (hls_binary == 1))] = 255
    # return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)  # debug code

    # 4) Perspective Transform
    binary_warped = cv2.warpPerspective(combined, M, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_NEAREST)
    # return cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)  # debug code
    # return cv2.cvtColor(cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)  # debug code

    # 5) Find Lanes via Sliding Windows: 1st Method

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set Searching Parameters
    margin = 80  # Set the width of the windows +/- margin
    minpix = 50  # Set minimum number of pixels found to recenter window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    print(left_fitx)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # return out_img  # debug code

    # # 5) Find Lanes via Sliding Windows: 2nd method
    #
    # window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin)
    # if len(window_centroids) > 0:
    #     # Points used to draw all the left and right windows
    #     l_points = np.zeros_like(binary_warped)
    #     r_points = np.zeros_like(binary_warped)
    #
    #     # Go through each level and draw the windows
    #     for level in range(0, len(window_centroids)):
    #         # Window_mask is a function to draw window areas
    #         l_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
    #         r_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)
    #         # Add graphic points from window mask here to total pixels found
    #         l_points[(l_points == 255) | ((l_mask == 1))] = 255
    #         r_points[(r_points == 255) | ((r_mask == 1))] = 255
    #
    #     # Draw the results: make left and right window pixels red and blue
    #     templateL = np.array(l_points, np.uint8)
    #     templateR = np.array(r_points, np.uint8)
    #     zero_channel = np.zeros_like(templateL)
    #     template = np.array(cv2.merge((templateR, zero_channel, templateL)), np.uint8)
    #     warpage = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)  # making the original road pixels 3 color channels
    #     output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
    #
    # else:
    #     # If no window centers found, just display orginal road image
    #     output = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8)
    #
    # # return output  # debug code
    #



    # Detect lane lines
    # -> https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/c41a4b6b-9e57-44e6-9df9-7e4e74a1a49a

    # Determine the lane curvature
    # filter curvature values

    # detect offset of the car position





    return out_img


    # Tracking -> tips

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



    return cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)
    return warped



######################################
# process frame by frame

# clip1 = VideoFileClip('../challenge_video.mp4')
# clip1 = VideoFileClip('../project_video.mp4')
# clip1 = VideoFileClip('../harder_challenge_video.mp4')

for file in ('../project_video.mp4', '../challenge_video.mp4', '../harder_challenge_video.mp4'):
    clip1 = VideoFileClip(file)
    frameno = 0
    for frame in clip1.iter_frames():
        if frameno % 10 == 0:
            print('frameno: {:5.0f}'.format(frameno))
            result = process_image(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = cv2.vconcat([cv2.resize(frame, (800, 380)),
                               cv2.resize(result, (800, 380))])
            # cv2.imshow('result', result)
            cv2.imshow('frame', img)
    
        frameno += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
exit(0)

######################################

white_output = '/Users/ichikihiroshi/CarND-Advanced-Lane-Lines/project_video_out.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
exit(0)


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
