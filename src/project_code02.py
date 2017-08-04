
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


def warper(img, src, dst):
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped



# source and destination points
# Given src and dst points, calculate the perspective transform matrix

# perspective_src = np.float32([[631, 425], [649, 425], [1055, 675], [265, 675]])
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]])  # sample data
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 700], [203, 700]])  # ignore bonnet
# perspective_src = np.float32([[582, 460], [698, 460], [1127, 695], [203, 695]])  # a little adjustment
# perspective_src = np.float32([[585, 460], [695, 460], [1127, 695], [203, 695]])  # a little adjustment
perspective_src = np.float32([[585, 460], [695, 460], [1127, 685], [203, 685]])  # prevent bonnnet
(width, height) = (1280, 720)
perspective_dst = np.float32([[320, 0], [width - 320, 0], [width - 320, height - 0], [320, height - 0]])


M = cv2.getPerspectiveTransform(perspective_src, perspective_dst)
Minv = cv2.getPerspectiveTransform(perspective_dst, perspective_src)

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


def create_binary_image(image):
    # Choose a Sobel kernel size

    # 2) Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # gray = gaussian_blur(gray, kernel_size=5)

    # Apply each of the thresholding functions
    # ksize = 15  # Choose a larger odd number to smooth gradient measurements
    # gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, sobel_thresh=(30, 100))  # 20, 80
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, sobel_thresh=(30, 100))  # 20, 80
    # dir_binary = dir_threshold(gray, sobel_kernel=7, thresh=(0.7, 1.3))  # doesn't work for harder_challenge_video

    # mag_binary = mag_thresh(gray, sobel_kernel=15, mag_thresh=(30, 100))  # 20, 100
    # hls_asphalt = hls_select(image, hthresh=(100, 200), ithresh=(0, 255), sthresh=(0, 255))  # Asphalt color

    # hls_yellow = hls_select(image, hthresh=(10, 30), ithresh=(50, 230), sthresh=(50, 255))  # yellow line all
    hls_yellow1 = hls_select(image, hthresh=(10, 30), ithresh=(50, 150), sthresh=(30, 255))  # yellow line dark
    hls_yellow2 = hls_select(image, hthresh=(20, 30), ithresh=(120, 250), sthresh=(30, 255))  # yellow line light

    rgb_white = rgb_select(image, rthresh=(200, 255), gthresh=(200, 255), bthresh=(200, 255))  # white line
    # rgb_white = rgb_select(image, rthresh=(190, 255), gthresh=(190, 255), bthresh=(190, 255))  # white line
    rgb_excess = rgb_select(image, rthresh=(250, 255), gthresh=(250, 255), bthresh=(250, 255))  # white line

    # combined = np.zeros_like(dir_binary)
    combined = np.zeros((rgb_white.shape), dtype=np.uint8)
    # # XX combined[((gradx == 1) | (grady == 1))] = 1  # shallow edge
    combined[((hls_yellow1 == 1) | (hls_yellow2 == 1))] = 1  # yellow line
    combined[((rgb_white == 1) & (rgb_excess != 1))] = 1  # White line
    # combined[((mag_binary == 1) & (hls_asphalt != 1))] = 1  # none Asphalt edge
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
            win_y_high = image.shape[0] - window * window_height  + window_height // 2
            win_xleft_low = leftx_current - int(margin * 1.5)
            win_xleft_high = leftx_current + int(margin * 1.5)
            win_xright_low = rightx_current - int(margin * 1.5)
            win_xright_high = rightx_current + int(margin * 1.5)
        else:
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height  + window_height // 2
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


left_fit = [0, 0, 360]
right_fit = [0, 0, 920]
left_curverad = 0
right_curverad = 0
pre_left_curverad = 0
pre_right_curverad = 0

def process_image(image, weight=0.5):

    # 1) Undistort using mtx and dist
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # return undist
    # return cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)  # debug code


    # 3) Create binary image via Combining Threshold
    combined = create_binary_image(undist)
    # combined *= 255
    # return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)  # debug code


    # 4) Perspective Transform
    binary_warped = cv2.warpPerspective(combined, M, (combined.shape[1], combined.shape[0]), flags=cv2.INTER_NEAREST)
    # binary_warped *= 255
    # return cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)  # debug code


    # 5) Find Lanes via Sliding Windows: 1st Method

    # 5-1) search lane candidates
    c_left_fit, c_right_fit, out_img = sliding_windows_search(binary_warped)

    # 5-2) Generate x and y values for pixel image
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = c_left_fit[0] * ploty**2 + c_left_fit[1] * ploty + c_left_fit[2]
    right_fitx = c_right_fit[0] * ploty**2 + c_right_fit[1] * ploty + c_right_fit[2]
    # Display
    if c_left_fit[2] != 0:
        for x, y in zip(left_fitx, ploty):
            cv2.circle(out_img, (int(x), int(y)), 1, color=[255, 200, 200], thickness=1)
    if c_right_fit[2] != 0:
        for x, y in zip(right_fitx, ploty):
            cv2.circle(out_img, (int(x), int(y)), 1, color=[200, 200, 255], thickness=1)
    # return out_img

    # 6) Determine the lane curvature
    global left_curverad, right_curverad

    # 6-1) Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.66 / 700 # meters per pixel in x dimension
    c_left_curverad = measure_curvature(left_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    c_right_curverad = measure_curvature(right_fitx, ploty, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)

    if left_curverad == 0:
        left_curverad = c_left_curverad
    if right_curverad == 0:
        right_curverad = c_right_curverad


    # 7) Sanity Check
    global left_fit, right_fit

    left_validity = True
    right_validity = True

    # 7-1) Check status of SlidingWindow function
    if c_left_fit[2] == 0:
        left_validity = False
    if c_right_fit[2] == 0:
        right_validity = False


    # 7-2) Checking that they have similar curvature
    global pre_left_curverad, pre_right_curverad

    diff_left_curverad = abs(pre_left_curverad - c_left_curverad) / left_curverad
    diff_right_curverad = abs(pre_right_curverad - c_right_curverad) / right_curverad

    diff_curverad_thresh = 5
    if pre_left_curverad != 0 and diff_curverad_thresh < diff_left_curverad:
        left_validity = False
    if pre_right_curverad != 0 and diff_curverad_thresh < diff_right_curverad:
        right_validity = False

    print('  diff_curv L:{:3.1f}%, R:{:3.1f}%'.format(diff_left_curverad, diff_right_curverad), end='')

    pre_left_curverad = left_curverad
    pre_right_curverad = right_curverad


    # 7-3) Checking that they are roughly parallel
    # - Checking that they are separated by approximately the right distance horizontally
    # - Checking that they are roughly parallel
    road_width = right_fitx[0] - left_fitx[0]
    road_width_1 = right_fitx[-1] - left_fitx[-1]
    road_width_m = right_fitx[len(right_fitx)//2] - left_fitx[len(left_fitx)//2]

    road_width_thresh_min = 300
    road_width_thresh_max = 1000
    if road_width < road_width_thresh_min or road_width_thresh_max < road_width:
        right_validity = False
        left_validity = False
    if road_width_1 < road_width_thresh_min or road_width_thresh_max < road_width_1:
        right_validity = False
        left_validity = False
    if road_width_m < road_width_thresh_min or road_width_thresh_max < road_width_m:
        right_validity = False
        left_validity = False

    print('  Road {:3.0f} {:3.0f} {:3.0f}'.format(road_width, road_width_m, road_width_1), end='')


    # 7-4) Update Fitting Data
    if c_left_fit[2] != 0 and left_validity:
        left_fit = (left_fit + c_left_fit) / 2
    if c_right_fit[2] != 0 and right_validity:
        right_fit = (right_fit + c_right_fit) / 2

    # Display latest Fitting Curves
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    for x, y in zip(left_fitx, ploty):
        cv2.circle(out_img, (int(x), int(y)), 1, color=[255, 100, 100], thickness=2)
    for x, y in zip(right_fitx, ploty):
        cv2.circle(out_img, (int(x), int(y)), 1, color=[100, 100, 255], thickness=2)

    # return out_img


    # 7-5) Determine Curvature Value
    if left_validity and left_validity:
        left_curverad = c_left_curverad
    if right_validity and right_validity:
        right_curverad = c_right_curverad

    print('  Curv {:5.1f} {:5.1f}'.format(left_curverad, right_curverad), end='')

    if not left_validity:
        print('  invalid-Left', end='')
    if not right_validity:
        print('  invalid-Right', end='')
    print()



    # 7-6) Detect offset of the car position
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
    vehicle_offset = 1280 / 2 - lane_center
    vehicle_offset *= xm_per_pix
    # print('  offset {:8.1f}m, curvature left:{:8.1f}m,  right:{:8.1f}m'.format(vehicle_offset, left_curverad, right_curverad))
    # Example values: 632.1 m    626.2 m


    # Display
    for lx, rx, y in zip(left_fitx, right_fitx, ploty):
        cv2.circle(out_img, (int(lx), int(y)), 1, color=[255, 200, 200], thickness=4)
        cv2.circle(out_img, (int(rx), int(y)), 1, color=[200, 200, 255], thickness=4)
    cv2.line(out_img, (int(lane_center), 0), (int(lane_center), 719), color=[200, 200, 200], thickness=4)
    cv2.line(out_img, (out_img.shape[1]//2, 0), (out_img.shape[1]//2, out_img.shape[0] - 1), color=[255, 255, 255], thickness=1)
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
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
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
    for file in ('project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4'):
        clip1 = VideoFileClip('../' + file)
        frameno = 0
        left_fit = [0, 0, 360]
        right_fit = [0, 0, 920]
        left_curverad = 0
        right_curverad = 0
        pre_left_curverad = 0
        pre_right_curverad = 0

        for frame in clip1.iter_frames():
            if frameno % 10 == 0 and frameno < 1000:
                # print('frameno: {:5.0f}'.format(frameno))
                result = process_image(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_lines(frame, trapezoid, color=[100, 100, 180], thickness=2)

                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                img = cv2.vconcat([cv2.resize(frame, (800, 380)),
                                   cv2.resize(result, (800, 380))])
                # cv2.imshow('result', result)
                cv2.imshow('frame', img)
                if frameno % 100 == 0:
                    name, ext = os.path.splitext(os.path.basename(file))
                    filename = '{}_{:04.0f}fr.jpg'.format(name, frameno)
                    if not os.path.exists(filename):
                        cv2.imwrite(filename, img)
            frameno += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
exit(0)

######################################

white_output = '../project_video_out.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

white_output = '../challenge_video_out.mp4'
clip1 = VideoFileClip('../challenge_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

white_output = '../harder_challenge_video_out.mp4'
clip1 = VideoFileClip('../harder_challenge_video.mp4')
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
