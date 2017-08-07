
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


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
import pickle
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


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


def hls_select(image, hthresh=(0, 255), sthresh=(0, 255), ithresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to each channel
    h_channel = hls[:, :, 0]
    i_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(((h_channel >= hthresh[0]) & (h_channel <= hthresh[1])) &
                   ((s_channel >= sthresh[0]) & (s_channel <= sthresh[1])) &
                   ((i_channel >= ithresh[0]) & (i_channel <= ithresh[1])))] = 1
    return binary_output

def rgb_select(image, rthresh=(0, 255), gthresh=(0, 255), bthresh=(0, 255)):
    b_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    r_channel = image[:, :, 2]
    binary_output = np.zeros_like(g_channel)
    binary_output[(((r_channel >= rthresh[0]) & (r_channel <= rthresh[1])) &
                   ((g_channel >= gthresh[0]) & (g_channel <= gthresh[1])) &
                   ((b_channel >= bthresh[0]) & (b_channel <= bthresh[1])))] = 1
    return binary_output


def warper(img, M):
    # Compute and apply perpective transform
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped



# source and destination points
# Given src and dst points, calculate the perspective transform matrix

# perspective_src = np.float32([[631, 425], [649, 425], [1055, 675], [265, 675]])  # trial
# perspective_src = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]])  # sample data
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 700], [203, 700]])  # ignore bonnet
# perspective_src = np.float32([[582, 460], [698, 460], [1127, 695], [203, 695]])  # a little adjustment
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 695], [203, 695]])  # a little adjustment
perspective_src = np.float32([[585, 460], [695, 460], [1127, 685], [203, 685]])  # prevent bonnnet
# perspective_src = np.float32([[600, 440], [640, 440], [1105, 675], [295, 675]])  # trial

(width, height) = (1280, 720)
perspective_dst = np.float32([[320, 0], [width - 320, 0], [width - 320, height - 0], [320, height - 0]])


# Calculate the Perspective Transformation Matrix and its invert Matrix
M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)


def create_binary_image_light(image):
    """
    output a binary image that show lane lines
    light algorithm
    """

    # 1) White Line
    rgb_white = rgb_select(image, rthresh=(200, 255), gthresh=(200, 255), bthresh=(200, 255))  # white line
    rgb_excess = rgb_select(image, rthresh=(250, 255), gthresh=(250, 255), bthresh=(250, 255))  # white line

    # 2) Yellow Line
    hls_yellow1 = hls_select(image, hthresh=(10, 30), ithresh=(50, 150), sthresh=(30, 255))  # yellow line dark
    hls_yellow2 = hls_select(image, hthresh=(20, 30), ithresh=(120, 250), sthresh=(30, 255))  # yellow line light

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((rgb_white.shape), dtype=np.uint8)
    combined[((hls_yellow1 == 1) | (hls_yellow2 == 1))] = 1  # yellow line
    combined[((rgb_white == 1) & (rgb_excess != 1))] = 1  # White line

    return combined


def create_binary_image(image):
    """
    output a binary image that show lane lines
    """
    # 1) Line Edge
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, sobel_thresh=(25, 50))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, sobel_thresh=(50, 150))
    mag_binary = mag_thresh(gray, sobel_kernel=15, mag_thresh=(50, 250))
    dir_binary = dir_threshold(gray, sobel_kernel=7, thresh=(0.7, 1.3))
    # hls_binary = hls_select(image, hthresh=(50, 100), ithresh=(0, 255), sthresh=(90, 190))  # Asphalt color
    hls_binary = hls_select(image, hthresh=(0, 255), ithresh=(0, 255), sthresh=(90, 190))  # Asphalt color

    # 2) Yellow Line
    hls_yellow1 = hls_select(image, hthresh=(10, 30), ithresh=(50, 150), sthresh=(30, 255))  # yellow line dark
    hls_yellow2 = hls_select(image, hthresh=(20, 30), ithresh=(120, 250), sthresh=(30, 255))  # yellow line light

    rgb_white = rgb_select(image, rthresh=(200, 255), gthresh=(200, 255), bthresh=(200, 255))  # white line
    rgb_excess = rgb_select(image, rthresh=(250, 255), gthresh=(250, 255), bthresh=(250, 255))  # white line

    # 3) Concrete
    hls_binary2 = hls_select(image, hthresh=(50, 100), ithresh=(0, 255), sthresh=(90, 190))  # shadow

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((mag_binary.shape), dtype=np.uint8)
    combined[((gradx == 1) | (grady == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)) & (hls_binary2 != 1)] = 1

    return combined

def get_base_position(image, pos='left'):
    """
    input:image must have binary values
    """
    # Get two base positions from the histgram
    if pos == 'left':
        base = 360
    else:
        base = 920

    # for div in (10, 9, 8, 7, 6, 5, 4, 3, 2):
    #     # Take a histogram of the bottom half of the image
    #     histogram = np.sum(image[image.shape[0] // div:, :], axis=0)
    #     # Find the peak of the left and right halves of the histogram
    #     # These will be the starting point for the left and right lines
    #     midpoint = np.int(histogram.shape[0] / 2)
    #     if pos == 'left':
    #         if 30 < np.max(histogram[:midpoint]):
    #             print('left base :', np.argmax(histogram[:midpoint]), '  with ', div, '/', midpoint)
    #             return np.argmax(histogram[:midpoint])
    #     else:
    #         if 30 < np.max(histogram[midpoint:]):
    #             # print('right base:', np.argmax(histogram[midpoint:]) + midpoint, '  with ', div, '/', midpoint)
    #             return np.argmax(histogram[midpoint:]) + midpoint

    return base


def sliding_windows_search(image):
    """
    input:image must have binary values
    """

    # Get two base positions from the histgram
    leftx_base = get_base_position(image, 'left')
    rightx_base = get_base_position(image, 'right')


    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_different = 0
    rightx_different = 0
    # Set Searching Parameters
    margin = 90  # Set the width of the windows +/- margin
    minpix = 70  # Set minimum number of pixels found to recenter window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_index = []
    right_lane_index = []

    # Step through the windows one by one
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        if window == 0:
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height + window_height // 2
            win_xleft_low = leftx_current - int(margin * 1.5)
            win_xleft_high = leftx_current + int(margin * 1.5)
            win_xright_low = rightx_current - int(margin * 1.5)
            win_xright_high = rightx_current + int(margin * 1.5)
        else:
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height + window_height // 2
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_index  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_index = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_index.append(good_left_index)
        right_lane_index.append(good_right_index)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_index) > minpix:
            leftx_candidate = np.int(np.mean(nonzerox[good_left_index]))
            leftx_different = leftx_candidate - leftx_current
            leftx_current = leftx_candidate
        else:
            leftx_current += leftx_different
        if len(good_right_index) > minpix:
            rightx_candidate = np.int(np.mean(nonzerox[good_right_index]))
            rightx_different = rightx_candidate - rightx_current
            rightx_current = rightx_candidate
        else:
            rightx_current += rightx_different

    # Concatenate the arrays of indices
    left_lane_index = np.concatenate(left_lane_index)
    right_lane_index = np.concatenate(right_lane_index)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_index]
    lefty = nonzeroy[left_lane_index]
    rightx = nonzerox[right_lane_index]
    righty = nonzeroy[right_lane_index]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except TypeError:
        # print('TypeError: left polyfit')
        left_fit = [0, 0, 0]
        # return [0, 0, leftx_base], [0, 0, rightx_base], out_img

    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        # print('TypeError: right polyfit')
        right_fit = [0, 0, 0]
        # return [0, 0, leftx_base], [0, 0, rightx_base], out_img

    out_img[nonzeroy[left_lane_index], nonzerox[left_lane_index]] = [255, 150, 150]
    out_img[nonzeroy[right_lane_index], nonzerox[right_lane_index]] = [150, 150, 255]

    return left_fit, right_fit, out_img


def measure_curvature(xs, ys, ym_per_pix, xm_per_pix):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ys * ym_per_pix, xs * xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ys)
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad


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


def sliding_windows_search_2nd(image):
    # window settings
    window_width = 50
    window_height = 80  # Break image into 9 vertical layers since image height is 720
    margin = 70  # How much to slide left and right for searching

    window_centroids = find_window_centroids(image, window_width, window_height, margin)
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results: make left and right window pixels red and blue
        templateL = np.array(l_points, np.uint8)
        templateR = np.array(r_points, np.uint8)
        zero_channel = np.zeros_like(templateL)
        template = np.array(cv2.merge((templateR, zero_channel, templateL)), np.uint8)
        warpage = np.array(cv2.merge((image, image, image)), np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    else:
        # If no window centers found, just display orginal road image
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output


def fit_quadratic_polynomial(c_left_fit, c_right_fit, image):
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = c_left_fit[0] * ploty**2 + c_left_fit[1] * ploty + c_left_fit[2]
    right_fitx = c_right_fit[0] * ploty**2 + c_right_fit[1] * ploty + c_right_fit[2]

    return left_fitx, right_fitx, ploty


def sanity_chack_polynomial(fit, pre_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200)):
    if fit[1] == 0:
        return False, pre_fit
    if pre_fit[1] == 0:
        return True, fit

    diff = pre_fit - fit
    
    if diff[0] < thresh0[0] or thresh0[1] < diff[0]:
        return False, fit
    if diff[1] < thresh1[0] or thresh1[1] < diff[1]:
        return False, fit
    if diff[2] < thresh2[0] or thresh2[1] < diff[2]:
        return False, fit

    return True, fit

def sanity_chack_curverad(left_curverad, right_curverad, diff_curverad_thresh=40):
    global pre_left_curverad, pre_right_curverad
    if pre_left_curverad == 0 or pre_right_curverad == 0:
        pre_left_curverad = left_curverad
        pre_right_curverad = right_curverad
        return True, True

    diff_left_curverad = 100 * abs(pre_left_curverad - left_curverad) / pre_left_curverad
    diff_right_curverad = 100 * abs(pre_right_curverad - right_curverad) / pre_right_curverad

    left_validity = True
    right_validity = True

    if diff_curverad_thresh < diff_left_curverad:
        left_validity = False
    if diff_curverad_thresh < diff_right_curverad:
        right_validity = False

    pre_left_curverad = left_curverad
    pre_right_curverad = right_curverad

    # print('  diff L:{:5.1f}%, R:{:5.1f}%'.format(diff_left_curverad, diff_right_curverad), end='')

    return left_validity, right_validity


def sanity_chack_roadwidth(left_fitx, right_fitx, thresh0=(300, 1200), thresh1=(300, 1200), thresh2=(300, 1200)):

    road_width0 = right_fitx[0] - left_fitx[0]
    road_width1 = right_fitx[len(right_fitx)//2] - left_fitx[len(left_fitx)//2]
    road_width2 = right_fitx[-1] - left_fitx[-1]

    # print('   Road {:4.0f} {:4.0f} {:4.0f}'.format(road_width0, road_width1, road_width2), end='')

    if road_width0 < thresh0[0] or thresh0[1] < road_width0:
        return False
    if road_width1 < thresh1[0] or thresh1[1] < road_width1:
        return False
    if road_width2 < thresh2[0] or thresh2[1] < road_width2:
        return False

    return True


def process_image(image, weight=0.5):

    # 1) Undistort using mtx and dist
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # return undist
    # return cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)  # debug code
    # undist = image

    # 3) Create binary image via Combining Threshold
    combined = create_binary_image_light(undist)
    # combined = create_binary_image(undist)
    # combined *= 255
    # return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)  # debug code


    # 4) Perspective Transform
    # binary_warped = cv2.warpPerspective(combined, M, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_NEAREST)
    binary_warped = warper(combined, M)
    # binary_warped *= 255
    # return cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)  # debug code


    # 5) Find Lanes via Sliding Windows: 2ndst Method
    # out_img = sliding_windows_search_2nd(binary_warped)
    # return out_img


    # 5) Find Lanes via Sliding Windows: 1st Method

    # 5-1) search lane candidates
    c_left_fit, c_right_fit, out_img = sliding_windows_search(binary_warped)

    # 5-2) Generate x and y values for pixel image
    left_fitx, right_fitx, ploty = fit_quadratic_polynomial(c_left_fit, c_right_fit, binary_warped)

    # 5-3) Check initial status of SlidingWindow function
    left_validity = True
    right_validity = True
    if c_left_fit[1] == 0:
        left_validity = False
    if c_right_fit[1] == 0:
        right_validity = False

    # Display
    # if c_left_fit[2] != 0:
    #     for x, y in zip(left_fitx, ploty):
    #         cv2.circle(out_img, (int(x), int(y)), 1, color=[255, 200, 200], thickness=1)
    # if c_right_fit[2] != 0:
    #     for x, y in zip(right_fitx, ploty):
    #         cv2.circle(out_img, (int(x), int(y)), 1, color=[200, 200, 255], thickness=1)

    # print('L:{:+.4f}, {:+.3f}, {:6.1f}  '.format(c_left_fit[0], c_left_fit[1], c_left_fit[2]), end='')
    # print('R:{:+.4f}, {:+.3f}, {:6.1f}'.format(c_right_fit[0], c_right_fit[1], c_right_fit[2]), end='')
    # return out_img



    # 6) Determine the lane curvature
    global left_fit, right_fit
    global pre_left_fit, pre_right_fit
    global left_curverad, right_curverad
    global pre_left_curverad, pre_right_curverad


    # 6-1) Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    c_left_curverad = measure_curvature(left_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    c_right_curverad = measure_curvature(right_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    if left_curverad == 0:
        left_curverad = c_left_curverad
    if right_curverad == 0:
        right_curverad = c_right_curverad

    # print('  Curve {:7.1f} {:7.1f}'.format(c_left_curverad, c_right_curverad), end='')


    # 7) Sanity Check
    # 7-1) Checking that they have stable coefficients
    c_left_validity, pre_left_fit = sanity_chack_polynomial(c_left_fit, pre_left_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200))
    c_right_validity, pre_right_fit = sanity_chack_polynomial(c_right_fit, pre_right_fit, thresh0=(-0.0007, 0.0007), thresh1=(-0.5, 0.5), thresh2=(-200, 200))
    if not c_left_validity:
        left_validity = False
    if not c_right_validity:
        right_validity = False


    # 7-2) Checking that they have similar curvature
    c_left_validity, c_right_validity = sanity_chack_curverad(c_left_curverad, c_right_curverad, diff_curverad_thresh=30)
    if not c_left_validity:
        left_validity = False
    if not c_right_validity:
        right_validity = False


    # 7-2) Checking that they are separated by approximately the right distance horizontally
    # 7-3) Checking that they are roughly parallel
    c_validity = sanity_chack_roadwidth(left_fitx, right_fitx, thresh0=(200, 1180), thresh1=(400, 1050), thresh2=(500, 780))  # 640 at horizontal road
    if left_validity and right_validity and not c_validity:
        right_validity = False
        left_validity = False




    # 7-4) Update Fitting Data
    if c_left_fit[2] != 0 and left_validity:
        left_fit = (left_fit + c_left_fit) / 2
    if c_right_fit[2] != 0 and right_validity:
        right_fit = (right_fit + c_right_fit) / 2
    # if left_validity:
    #     left_fit = c_left_fit
    # if right_validity:
    #     right_fit = c_right_fit

    # print('  poly ', end='')
    # print('L:{:+.4f}, {:+.3f}, {:6.1f}  '.format(left_fit[0], left_fit[1], left_fit[2]), end='')
    # print('R:{:+.4f}, {:+.3f}, {:6.1f}'.format(right_fit[0], right_fit[1], right_fit[2]), end='')

    # Display latest Fitting Curves
    # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    # for x, y in zip(left_fitx, ploty):
    #     cv2.circle(out_img, (int(x), int(y)), 1, color=[255, 100, 100], thickness=2)
    # for x, y in zip(right_fitx, ploty):
    #     cv2.circle(out_img, (int(x), int(y)), 1, color=[100, 100, 255], thickness=2)
    # return out_img


    # 7-5) Determine Curvature Value
    if left_curverad == 0 or left_validity:
        left_curverad = c_left_curverad
    if right_curverad == 0 or right_validity:
        right_curverad = c_right_curverad

    # print('  Curv {:6.1f} {:6.1f}'.format(left_curverad, right_curverad), end='')



    # 7-6) Detect offset of the car position
    left_fitx, right_fitx, ploty = fit_quadratic_polynomial(left_fit, right_fit, binary_warped)
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    vehicle_offset = 1280 / 2 - lane_center
    vehicle_offset *= xm_per_pix

    # print('  offset {:8.1f}m, curvature left:{:8.1f}m,  right:{:8.1f}m'.format(vehicle_offset, left_curverad, right_curverad), end='')
    # Example values: 632.1 m    626.2 m

    # if not left_validity or not right_validity:
    #     print('  invalid:', end='')
    # if not left_validity:
    #     print(' Left', end='')
    # if not right_validity:
    #     print(' Right', end='')
    # print()


    # Display
    # for lx, rx, y in zip(left_fitx, right_fitx, ploty):
    #     cv2.circle(out_img, (int(lx), int(y)), 1, color=[255, 200, 200], thickness=4)
    #     cv2.circle(out_img, (int(rx), int(y)), 1, color=[200, 200, 255], thickness=4)
    # cv2.line(out_img, (int(lane_center), 0), (int(lane_center), 719), color=[200, 200, 200], thickness=4)
    # cv2.line(out_img, (out_img.shape[1]//2, 0), (out_img.shape[1]//2, out_img.shape[0] - 1), color=[255, 255, 255], thickness=1)
    # return out_img  # debug code




    # Drawing 

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    center_fitx = (right_fitx + left_fitx) / 2
    for x, y in zip(center_fitx, ploty):
        cv2.circle(color_warp, (int(x), int(y)), 1, color=[255, 255, 255], thickness=8)



    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    newwarp = warper(color_warp, Minv)

    # Combine the result with the original image
    font_size = 1.2
    font = cv2.FONT_HERSHEY_DUPLEX
    # infotext = 'offset {:+4.1f}m, curvature left:{:.1f}m,  right:{:.1f}m'.format(vehicle_offset, left_curverad, right_curverad)
    if vehicle_offset < 0:
        infotext = 'car position {:4.2f}m left , curvature {:.1f}m'.format(-vehicle_offset, (left_curverad + right_curverad)/2)
    elif vehicle_offset > 0:
        infotext = 'car position {:4.2f}m right, curvature {:.1f}m'.format(vehicle_offset, (left_curverad + right_curverad)/2)
    else:
        infotext = 'car position        center, curvature {:.1f}m'.format((left_curverad + right_curverad)/2)
    cv2.putText(undist, infotext, (30, 50), font, font_size, (255,255,255))

    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


######################################
# process frame by frame for developing


trapezoid = []
trapezoid.append([[perspective_src[0][0], perspective_src[0][1], perspective_src[1][0], perspective_src[1][1]]])
trapezoid.append([[perspective_src[1][0], perspective_src[1][1], perspective_src[2][0], perspective_src[2][1]]])
trapezoid.append([[perspective_src[2][0], perspective_src[2][1], perspective_src[3][0], perspective_src[3][1]]])
trapezoid.append([[perspective_src[3][0], perspective_src[3][1], perspective_src[0][0], perspective_src[0][1]]])

for l in range(1, 20):
    # for file in ('challenge_video.mp4', 'project_video.mp4', 'harder_challenge_video.mp4'):
    # for file in ('project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4'):
    # for file in ('challenge_video.mp4', 'project_video.mp4', 'harder_challenge_video.mp4'):
    for file in ('challenge_video.mp4', 'project_video.mp4'):
        clip1 = VideoFileClip('../' + file)

        left_fit = [0, 0, 360]
        right_fit = [0, 0, 920]
        pre_left_fit = [0, 0, 0]
        pre_right_fit = [0, 0, 0]

        left_curverad = 0
        right_curverad = 0
        pre_left_curverad = 0
        pre_right_curverad = 0

        frameno = 0
        for frame in clip1.iter_frames():
            # if frameno % 1 == 0 and ((520 <= frameno and frameno < 620) or (950 <= frameno and frameno < 1100)):
            # if frameno % 2 == 0 and ((530 <= frameno and frameno < 620) or (950 <= frameno and frameno < 1100)):
            # if 510 <= frameno and frameno < 590 and frameno % 5 == 0 and frameno < 640:
            if frameno % 1 == 0 and frameno < 1500:
                # print('frameno: {:5.0f}'.format(frameno))
                result = process_image(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_lines(frame, trapezoid, color=[100, 100, 180], thickness=2)

                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                img = cv2.vconcat([cv2.resize(frame, (800, 380)),
                                   cv2.resize(result, (800, 380))])

                # cv2.imshow('result', result)
                cv2.imshow('frame', img)

                # if frameno % 100 == 0:
                #     name, ext = os.path.splitext(os.path.basename(file))
                #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
                #     if not os.path.exists(filename):
                #         cv2.imwrite(filename, img)
                # if frameno == 300:
                #     name, ext = os.path.splitext(os.path.basename(file))
                #     filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
                #     cv2.imwrite(filename, img)
                #     exit(0)
            frameno += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
