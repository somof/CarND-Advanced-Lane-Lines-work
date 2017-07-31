

#4 Distortion

https://youtu.be/Bv3o0INBQIU

Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; 
this transformation isn’t perfect. 
Distortion actually changes what the shape and size of these 3D objects appear to be. 
So, the first step in analyzing camera images, 
is to undo this distortion so that you can get correct and useful information out of them.


# Camera Model

## Types of Distortion

Real cameras use curved lenses to form an image, 
and light rays often bend a little too much or too little at the edges of these lenses. 
This creates an effect that distorts the edges of images, 
so that lines or objects appear more or less curved than they actually are. 

This is called radial distortion, and it’s the most common type of distortion.

Another type of distortion, is tangential distortion. 
This occurs when a camera’s lens is not aligned perfectly parallel to the imaging plane, 
where the camera film or sensor is. 
This makes an image look tilted so that some objects appear farther away or closer than they actually are.
Distortion Coefficients and Correction

There are three coefficients needed to correct for radial distortion: k1, k2, and k3. 
To correct the appearance of radially distorted points in an image, one can use a correction formula.

In the following equations, (x,y) is a point in a distorted image. 
To undistort these points, OpenCV calculates r, which is the known distance between a point in an undistorted (corrected) image
 (x ​corrected​​ ,y ​corrected) and the center of the image distortion, which is often the center of that image (x ​c​​ ,y ​c​​ ). 
This center point (x c​​ ,y ​c​​ ) is sometimes referred to as the distortion center. These points are pictured below.

Note: The distortion coefficient k3 is required to accurately reflect major radial distortion (like in wide angle lenses). However, for minor radial distortion, which most regular camera lenses have, k3 has a value close to or equal to zero and is negligible. So, in OpenCV, you can choose to ignore this coefficient; this is why it appears at the end of the distortion values array: [k1, k2, p1, p2, k3]. In this course, we will use it in all calibration calculations so that our calculations apply to a wider variety of lenses (wider, like wide angle, haha) and can correct for both minor and major radial distortion.
Points in a distorted and undistorted (corrected) image. The point (x, y) is a single point in a distorted image and (x_corrected, y_corrected) is where that point will appear in the undistorted (corrected) image.
Points in a distorted and undistorted (corrected) image. The point (x, y) is a single point in a distorted image and (x_corrected, y_corrected) is where that point will appear in the undistorted (corrected) image.

Radial distortion correction.

There are two more coefficients that account for tangential distortion: p1 and p2, and this distortion can be corrected using a different correction formula.

Tangential distortion correction.


QUIZ QUESTION

Fun fact: the "pinhole camera" is not just a model, but was actually the earliest means of projecting images, first documented almost 2500 years ago in China!

Now back to computer vision... what is the fundamental difference between images formed with a pinhole camera and those formed using lenses?






#9 Finding Corners

In this exercise, you'll use the OpenCV functions 
findChessboardCorners() and drawChessboardCorners() 
to automatically find and draw corners in an image of a chessboard pattern.

To learn more about both of those functions, you can have a look at the OpenCV documentation here:
  cv2.findChessboardCorners() and cv2.drawChessboardCorners().

Applying these two functions to a sample image, you'll get a result like this:

In the following exercise, your job is simple. 
Count the number of corners in any given row and enter that value in nx. 
Similarly, count the number of corners in a given column and store that in ny. 
Keep in mind that "corners" are only points where two black and two white squares intersect, 
in other words, only count inside corners, not outside corners.

QUIZ
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # prepare object points
    nx = 0#TODO: enter the number of inside corners in x
    ny = 0#TODO: enter the number of inside corners in y
    
    # Make a list of calibration images
    fname = 'calibration_test.png'
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)


#10 Calibrating Your Camera

Examples of Useful Code

Converting an image, imported by cv2 or the glob API, to grayscale:

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

Note: If you are reading in an image using mpimg.imread() this will 
read in an RGB image and you should convert to grayscale using cv2.COLOR_RGB2GRAY, 
but if you are using cv2.imread() or the glob API, 
as happens in this video example, this will read in a BGR image 
and you should convert to grayscale using cv2.COLOR_BGR2GRAY. 

We'll learn more about color conversions later on in this lesson, 
but please keep this in mind as you write your own code and look at code examples.

Finding chessboard corners (for an 8x6 board):

    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

Drawing detected corners on an image:

    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)

Camera calibration, given object points, image points, and the shape of the grayscale image:

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

Undistorting a test image:

    dst = cv2.undistort(img, mtx, dist, None, mtx)

A note on image shape

The shape of the image, which is passed into the calibrateCamera function, 
is just the height and width of the image. 
One way to retrieve these values is by retrieving them from the grayscale image 
shape array gray.shape[::-1]. This returns the image height and width in pixel values like (960, 1280).

Another way to retrieve the image shape, 
is to get them directly from the color image by retrieving the first two values in the color image shape array 
using img.shape[0:2]. 
This code snippet asks for just the first two values in the shape array.

It's important to use an entire grayscale image shape or the first two values of a color image shape. 
This is because the entire shape of a color image will include a third value
 -- the number of color channels -- in addition to the height and width of the image. 

For example the shape array of a color image might be (960, 1280, 3), 
which are the pixel height and width of an image (960, 1280) and 
a third value (3) that represents the three color channels in the color image which you'll learn more about later, 
and if you try to pass these three values into the calibrateCamera function, you'll get an error.



#11 Correcting for Distortion

Here, you'll get a chance to try camera calibration and distortion correction for yourself!

There are two main steps to this process: use chessboard images to image points and object points, 
and then use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and undistortion.

Unfortunately, we can't perform the extraction of object points and image points in the browser quiz editor, 
so we provide these for you in the quiz below.

Try computing the calibration and undistortion in the exercise below, 
and if you want to play with extracting object points and image points yourself, 
fork the Jupyter notebook and images in this repository.

If you run into any errors as you run your code, 
please refer to the Examples of Useful Code section in the previous video and make sure that your code syntax matches up!


QUIZ

    import pickle
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    
    # Read in an image
    img = cv2.imread('test_image.png')
    
    # TODO: Write a function that takes an image, object points, and image points
    # performs the camera calibration, image distortion correction and 
    # returns the undistorted image
    def cal_undistort(img, objpoints, imgpoints):
        # Use cv2.calibrateCamera() and cv2.undistort()
        undist = np.copy(img)  # Delete this line
        return undist
    
    undistorted = cal_undistort(img, objpoints, imgpoints)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


Challenge: Calibrate your own camera!

If you're up for a challenge, go ahead and try these steps on your own camera images. 
Just print out a chessboard pattern, stick it to a flat surface, and take 20 or 30 pictures of it. 
Make sure to take pictures of the pattern over the entire field of view of your camera, particularly near the edges.

To extract object points and image points you can check out the Jupyter notebook in this repository. 
If you're following along with the code in the notebook, be sure you set the right size for your chessboard, 
the link above is to a 9 x 6 pattern, while before we were using an 8 x 6.

    git clone https://github.com/udacity/CarND-Camera-Calibration.git


#12 Lane Curvature

Calculating Lane Curvature

Self-driving cars need to be told the correct steering angle to turn, left or right. 
You can calculate this angle if you know a few things about the speed 
and dynamics of the car and how much the lane is curving.

One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, 
and from this you can easily extract useful information.

For a lane line that is close to vertical, 
you can fit a line using this formula:

    f(y) = Ay^2 + By + C, where A, B, and C are coefficients.

A gives you the curvature of the lane line, B gives you the heading or direction that the line is pointing, 
and C gives you the position of the line based on how far away it is from the very left of an image (y = 0).




# Perspective Transform

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective transform you’ll be most interested in is a bird’s-eye view transform that let’s us view a lane from above; this will be useful for calculating the lane curvature later on. Aside from creating a bird’s eye view representation of an image, a perspective transform can also be used for all kinds of different view points.



QUIZ QUESTION

There are various reasons you might want to perform a perspective transform on an image, 
but in the case of road images taken from our front-facing camera on a car, 
why are we interested in doing a perspective transform?



#15 Transform a Stop Sign

Examples of Useful Code

Compute the perspective transform, M, given source and destination points:

    M = cv2.getPerspectiveTransform(src, dst)

Compute the inverse perspective transform:

    Minv = cv2.getPerspectiveTransform(dst, src)

Warp an image using the perspective transform, M:

warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

Note: When you apply a perspective transform, choosing four source points manually, 
as we did in this video, is often not the best option. 
There are many other ways to select source points. 
For example, many perspective transform algorithms will programmatically detect four source points 
in an image based on edge or corner detection and analyzing attributes like color and surrounding pixels.




# Intuitions

QUIZ QUESTION

Here's a matching challenge to test your intuition about camera calibration, distortion, 
and perspective transforms. 
Drag the answers to the appropriate location in the column on the right.


# Undistort and Transform Perspective

Here's a tricky quiz for you! You have now seen how to find corners, 
calibrate your camera, undistort an image, and apply a perspective transform. 
Now it's your chance to perform all these steps on an image. 
In the last quiz you calibrated the camera, so here I'm giving you the camera matrix, mtx, 
and the distortion coefficients dist to start with.

Your goal is to generate output like the image shown above. 
To do that, you need to write a function that takes your distorted image as input and completes the following steps:

  - Undistort the image using cv2.undistort() with mtx and dist
  - Convert to grayscale
  - Find the chessboard corners
  - Draw corners
  - Define 4 source points (the outer 4 corners detected in the chessboard pattern)
  - Define 4 destination points (must be listed in the same order as src points!)
  - Use cv2.getPerspectiveTransform() to get M, the transform matrix
  - use cv2.warpPerspective() to apply M and warp your image to a top-down view

HINT: Source points are the x and y pixel values of any four corners on your chessboard, 
you can extract these from the corners array output from cv2.findChessboardCorners(). 
Your destination points are the x and y pixel values of where you want those four corners to be mapped to in the output image.
If you run into any errors as you run your code, 
please refer to the Examples of Useful Code section in the previous video and make sure that your code syntax matches up! 
For this example, please also refer back to the examples in the Calibrating Your Camera video.

    import pickle
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    # Read in an image
    img = cv2.imread('test_image2.png')
    nx = 8 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    
    # MODIFY THIS FUNCTION TO GENERATE OUTPUT 
    # THAT LOOKS LIKE THE IMAGE ABOVE
    def corners_unwarp(img, nx, ny, mtx, dist):
        # Pass in your image into this function
        # Write code to do the following steps
        # 1) Undistort using mtx and dist
        # 2) Convert to grayscale
        # 3) Find the chessboard corners
        # 4) If corners found: 
                # a) draw corners
                # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                     #Note: you could pick any four of the detected corners 
                     # as long as those four corners define a rectangle
                     #One especially smart way to do this would be to use four well-chosen
                     # corners that were automatically detected during the undistortion steps
                     #We recommend using the automatic detection of corners in your code
                # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
                # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
                # e) use cv2.warpPerspective() to warp your image to a top-down view
        #delete the next two lines
        M = None
        warped = np.copy(img) 
        return warped, M
    
    top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



# How I Did It

Solution to Undistort and Transform quiz:
That was a tricky quiz!

Let's take a closer look at the pieces involved. These steps will all be useful when you get to the project at the end of this section.

First off, we defined a function for you, corners_unwarp(). The function accepts an image and our previously calculated values for the camera matrix, mtx, and distortion coefficients, dist.

Then you had to implement the function.

Here's how I implemented mine:



    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients
    def corners_unwarp(img, nx, ny, mtx, dist):
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        if ret == True:
            # If we found corners, draw them! (just for fun)
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])
    
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)
    
        # Return the resulting image and matrix
        return warped, M



# Sobel Operator

The Sobel operator is at the heart of the Canny edge detection algorithm you used in the Introductory Lesson. 
Applying the Sobel operator to an image is a way of taking the derivative of the image in the x or y direction. 

The operators for Sobel_​x ​​  and Sobel_​y​​ , respectively, look like this:

These are examples of Sobel operators with a kernel size of 3 (implying a 3 x 3 operator in each case). 
This is the minimum size, but the kernel size can be any odd number. 
A larger kernel implies taking the gradient over a larger region of the image, or, in other words, a smoother gradient.

To understand how these operators take the derivative, you can think of overlaying either one on a 3 x 3 region of an image. 
If the image is flat across that region, then the result 
(summing the element-wise product of the operator and corresponding image pixels) will be zero. 

If, instead, for example, you apply the S_​x operator to a region of the image where values are rising from left to right, 
then the result will be positive, implying a positive derivative.

Derivative Example

If we apply the Sobel x and y operators to this image:

And then we take the absolute value, we get the result:

Absolute value of Sobel x (left) and Sobel y (right).
Absolute value of Sobel x (left) and Sobel y (right).

x vs. y

In the above images, you can see that the gradients taken in both the x and the y directions 
detect the lane lines and pick up other edges. 
Taking the gradient in the x direction emphasizes edges closer to vertical. 
Alternatively, taking the gradient in the y direction emphasizes edges closer to horizontal.

In the upcoming exercises, you'll write functions to take various thresholds of the x and y gradients. Here's some code that might be useful:


Examples of Useful Code

You need to pass a single color channel to the cv2.Sobel() function, so first convert it to grayscale:

gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

Note: Make sure you use the correct grayscale conversion depending on how you've read in your images. 
Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().

Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

Calculate the absolute value of the x derivative:

    abs_sobelx = np.absolute(sobelx)

Convert the absolute value image to 8-bit:

scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

Note: It's not entirely necessary to convert to 8-bit (range from 0 to 255) but in practice, 
it can be useful in the event that you've written a function to apply a particular threshold, 
and you want it to work the same on input images of different scales, like jpg vs. png. 
You could just as well choose a different standard range of values, like 0 to 1 etc.

Create a binary threshold to select pixels based on gradient strength:

    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    plt.imshow(sxbinary, cmap='gray')



# Applying Sobel

Here's your chance to write a function that will be useful for the Advanced Lane-Finding Project 
at the end of this lesson! Your goal in this exercise is to identify pixels where the gradient of 
an image falls within a specified threshold range.

Example

Here's the scaffolding for your function:

    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        # Grayscale
        # Apply cv2.Sobel()
        # Take the absolute value of the output from cv2.Sobel()
        # Scale the result to an 8-bit range (0-255)
        # Apply lower and upper thresholds
        # Create binary_output
        return binary_output

Pass in img and set the parameter orient as 'x' or 'y' to take either the x or y gradient. 
Set min_thresh, and max_thresh to specify the range to select for binary output. 
You can use exclusive (<, >) or inclusive (<=, >=) thresholding.

NOTE: Your output should be an array of the same size as the input image. 
The output array elements should be 1 where gradients were in the threshold range, and 0 everywhere else.
As usual, if you run into any errors as you run your code, please refer to the Examples of 
Useful Code section in the previous video and make sure that your code syntax matches up!


    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pickle
    
    
    # Read in an image and grayscale it
    image = mpimg.imread('signs_vehicles_xygrad.png')
    
    # Define a function that applies Sobel x or y, 
    # then takes an absolute value and applies a threshold.
    # Note: calling your function with orient='x', thresh_min=5, thresh_max=100
    # should produce output like the example image shown above this quiz.
    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        # 3) Take the absolute value of the derivative or gradient
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        # 6) Return this mask as your binary_output image
        binary_output = np.copy(img) # Remove this line
        return binary_output
        
    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
        # Return the result
        return binary_output



# Magnitude of the Gradient

With the result of the last quiz, you can now take the gradient in x or y 
and set thresholds to identify pixels within a certain gradient range. 
If you play around with the thresholds a bit, 
you'll find the x-gradient does a cleaner job of picking up the lane lines, 
but you can see the lines in the y-gradient as well.

In this next exercise, 
your goal is to apply a threshold to the overall magnitude of the gradient, 
in both x and y.

The magnitude, or absolute value, of the gradient is just 
the square root of the squares of the individual x and y gradients. 
For a gradient in both the x and y directions, 
the magnitude is the square root of the sum of the squares.

  abs_sobelx=√​(sobel_​x)^​2
  abs_sobely=√​(sobel_​y)^​2
  abs_sobelxy=√​((sobel_​x)^​2+(sobel_​y)^​2)

It's also worth considering the size of the region in the image over 
which you'll be taking the gradient. You can modify the kernel size for 
the Sobel operator to change the size of this region. 
Taking the gradient over larger regions can smooth over noisy intensity fluctuations 
on small scales. 
The default Sobel kernel size is 3, 
but here you'll define a new function that takes kernel size as a parameter 
(must be an odd number!)

The function you'll define for the exercise below should take in an image 
and optional Sobel kernel size, as well as thresholds for gradient magnitude. 
Next, you'll compute the gradient magnitude, apply a threshold, 
and create a binary output image showing where thresholds were met.

Steps to take in this exercise:

1. Fill out the function in the editor below to return a thresholded gradient magnitude. 
   Again, you can apply exclusive (<, >) or inclusive (<=, >=) thresholds.
2. Test that your function returns output similar to the example below for 
    sobel_kernel=9, mag_thresh=(30, 100).


    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pickle
    
    
    # Read in an image
    image = mpimg.imread('signs_vehicles_xygrad.png')
    
    # Define a function that applies Sobel x and y, 
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Calculate the magnitude 
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        # 5) Create a binary mask where mag thresholds are met
        # 6) Return this mask as your binary_output image

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
        # Return the binary image
        return binary_output
        
    # Run the function
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Direction of the Gradient

Steps to take in this exercise:

1. Fill out the function in the editor below to return a thresholded absolute value of the gradient direction.
   Use Boolean operators, again with exclusive (<, >) or inclusive (<=, >=) thresholds.
2. Test that your function returns output similar to the example below for
   sobel_kernel=15, thresh=(0.7, 1.3).

    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pickle
    
    
    # Read in an image
    image = mpimg.imread('signs_vehicles_xygrad.png')
    
    # Define a function that applies Sobel x and y, 
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
        
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image
    
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
        # Return the binary image
        return binary_output
        
    # Run the function
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dir_binary, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    



# Combining Thresholds

If you play around with the thresholds in the last exercise, 
you'll find that you can start to identify the lane lines by 
gradient direction alone by setting the threshold around thresh = (0.7, 1.3), 
but there's still a lot of noise in the resulting image.

Now consider how you can use various aspects of your gradient 
measurements (x, y, magnitude, direction) to isolate lane-line pixels. 
Specifically, think about how you can use thresholds of the x and y gradients, 
the overall gradient magnitude, 
and the gradient direction to focus on pixels that are likely to be part of the lane lines.

Challenge:
In the project at the end of this section, 
you'll want to experiment with thresholding various aspects of the gradient, 
so now would be a great time to start coding it up on your local machine! Grab the image we've been working with for the last three quizzes here (or a smaller jpg file here).

Combine the selection thresholds from the last 3 quizzes to write a piece of code like the following, 
where you can play with various thresholds and see the output.

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    return dir_binary

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
Try different combinations and see what you get.

For example, here is a selection for pixels where both the x and y gradients meet the threshold criteria, or the gradient magnitude and direction are both within their threshold values.

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
Output
Here is an example of a binary result from multiple thresholds:








# Reviewing Steps

Project Steps

Steps we’ve covered so far:

- Camera calibration
- Distortion correction
- Color/gradient threshold
- Perspective transform

After doing these steps, you’ll be given two additional steps for the project:

1. Detect lane lines
2. Determine the lane curvature




# Processing Each Image

In the project at the end of this module, the first thing you'll do is to compute the camera calibration matrix and distortion coefficients. You only need to compute these once, and then you'll apply them to undistort each new frame. Next, you'll apply thresholds to create a binary image and then apply a perspective transform.

Thresholding
You'll want to try out various combinations of color and gradient thresholds to generate a binary image where the lane lines are clearly visible. There's more than one way to achieve a good result, but for example, given the image above, the output you're going for should look something like this:

Perspective Transform
Next, you want to identify four source points for your perspective transform. In this case, you can assume the road is a flat plane. This isn't strictly true, but it can serve as an approximation for this project. You would like to pick four points in a trapezoidal shape (similar to region masking) that would represent a rectangle when looking down on the road from above.

The easiest way to do this is to investigate an image where the lane lines are straight, and find four points lying along the lines that, after perspective transform, make the lines look straight and vertical from a bird's eye view perspective.

Here's an example of the result you are going for with straight lane lines:

Now for curved lines
Those same four source points will now work to transform any image (again, under the assumption that the road is flat and the camera perspective hasn't changed). When applying the transform to new images, the test of whether or not you got the transform correct, is that the lane lines should appear parallel in the warped images, whether they are straight or curved.

Here's an example of applying a perspective transform to your thresholded binary image, using the same source and destination points as above, showing that the curved lines are (more or less) parallel in the transformed image:




# Finding the Lines

Locate the Lane Lines and Fit a Polynomial
Thresholded and perspective transformed image
Thresholded and perspective transformed image
You now have a thresholded warped image and you're ready to map out the lane lines! There are many ways you could go about this, but here's one example of how you might do it:

Line Finding Method: Peaks in a Histogram

After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image like this:

import numpy as np
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
The result looks like this:

Sliding Window
With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

Here is a short animation showing this method:

Implement Sliding Windows and Fit a Polynomial
Suppose you've got a warped binary image called binary_warped and you want to find which "hot" pixels are associated with the lane lines. Here's a basic implementation of the method shown in the animation above. You should think about how you could improve this implementation to make sure you can find the lines as robustly as possible!

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
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
Visualization
At this point, you're done! But here is how you can visualize the result as well:

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
The output should look something like this:

Skip the sliding windows step once you know where the lines are
Now you know where the lines are you have a fit! In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous line position like this:

# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
And you're done! But let's visualize the result here as well
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
And the output should look something like this:

The green shaded area shows where we searched for the lines this time. So, once you know where the lines are in one frame of video, you can do a highly targeted search for them in the next frame. This is equivalent to using a customized region of interest for each frame of video, and should help you track the lanes through sharp curves and tricky conditions. If you lose track of the lines, go back to your sliding windows search or other method to rediscover them.




# Sliding Window Search
Another way to approach the sliding window method is to apply a convolution, which will maximize the number of "hot" pixels in each window. A convolution is the summation of the product of two separate signals, in our case the window template and the vertical slice of the pixel image.

You slide your window template across the image from left to right and any overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the most likely position for the lane marker.

Now let's try using convolutions to find the best window center positions in a thresholded road image. The code below allows you to experiment with using convolutions for a sliding window search function. Go ahead and give it a try.


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import glob
    import cv2
    
    # Read in a thresholded image
    warped = mpimg.imread('warped_example.jpg')
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching
    
    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output
    
    def find_window_centroids(image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
    	    # convolve the window into the vertical slice of the image
    	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
    	    conv_signal = np.convolve(window, image_layer)
    	    # Find the best left centroid by using past left center as a reference
    	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    	    offset = window_width/2
    	    l_min_index = int(max(l_center+offset-margin,0))
    	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
    	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    	    # Find the best right centroid by using past right center as a reference
    	    r_min_index = int(max(r_center+offset-margin,0))
    	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
    	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    	    # Add what we found for that layer
    	    window_centroids.append((l_center,r_center))
    
        return window_centroids
    
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    
    # If we found any window centers
    if len(window_centroids) > 0:
    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    
    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()




# Measuring Curvature
You're getting very close to a final result! You have a thresholded image, where you've estimated which pixels belong to the left and right lane lines (shown in red and blue, respectively, below), and you've fit a polynomial to those pixel positions. Next we'll compute the radius of curvature of the fit.
Here I have fit the left and right lines with a second order polynomial shown in green.
Here I have fit the left and right lines with a second order polynomial shown in green.
In the last exercise, you located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve:

f(y)=Ay
​2
​​ +By+C

You're fitting for f(y), rather than f(x), because the lane lines in the warped image are near vertical and may have the same x value for more than one y value.

Radius of Curvature
The radius of curvature (awesome tutorial here) at any point x of the function x=f(y) is given as follows:

R
​curve
​​ =
​∣
​dy
​2
​​ 
​
​d
​2
​​ x
​​ ∣
​
​[1+(
​dy
​
​dx
​​ )
​2
​​ ]
​3/2
​​ 
​​ 

In the case of the second order polynomial above, the first and second derivatives are:

f
​′
​​ (y)=
​dy
​
​dx
​​ =2Ay+B

f
​′′
​​ (y)=
​dy
​2
​​ 
​
​d
​2
​​ x
​​ =2A

So, our equation for radius of curvature becomes:

R
​curve
​​ =
​∣2A∣
​
​(1+(2Ay+B)
​2
​​ )
​3/2
​​ 
​​ 

The y values of your image increase from top to bottom, so if, for example, you wanted to measure the radius of curvature closest to your vehicle, you could evaluate the formula above at the y value corresponding to the bottom of your image, or in Python, at yvalue = image.shape[0].
Sample Implementation
import numpy as np
import matplotlib.pyplot as plt
# Generate some fake data to represent lane-line pixels
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
The output looks like this:

Now we have polynomial fits and we can calculate the radius of curvature as follows:


# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)
# Example values: 1926.74 1908.48
But now we need to stop and think... We've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So we actually need to repeat this calculation after converting our x and y values to real world space.

This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.

So here's a way to repeat the calculation of radius of curvature after correcting for scale in x and y:

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 632.1 m    626.2 m
Check out the U.S. government specifications for highway curvature to see how your numbers compare. There's no need to worry about absolute accuracy in this case, but your results should be "order of magnitude" correct.




# Tips and Tricks for the Project
In this lesson, you've acquired some new tools to help you find and track the lane lines. By all means, you are welcome and encouraged to use the techniques you used in the very first project. Here are some tips for the upcoming project.

Camera Calibration
The calibration images in the lesson exercise were taken with a different camera setting and a different chessboard pattern than the calibration images for the project. You need to set your chessboard size to 9x6 for the project instead of 8x6 as in the lesson.

Do your curvature values make sense?
We're not expecting anything like perfection for this project, but a good check on whether or not your perspective transform worked as expected, your conversion from pixel space to world space was correct, and that you successfully calculated the radius of curvature is whether or not your results are roughly consistent with reality.

Here is an image from Google maps of where the project video was made (just northwest of the Udacity office!). Here, I've drawn a circle to coincide with the first left curve in the project video. This is a very rough estimate, but as you can see, the radius of that circle is approximately 1 km. You don't need to tune your algorithm to report exactly a radius of 1 km in the project, but if you're reporting 10 km or 0.1 km, you know there might be something wrong with your calculations!

Here are some other tips and tricks for building a robust pipeline:

Offset
You can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane.

Tracking
After you've tuned your pipeline on test images, you'll run on a video stream, just like in the first project. In this case, however, you're going to keep track of things like where your last several detections of the lane lines were and what the curvature was, so you can properly treat new detections. To do this, it's useful to define a Line() class to keep track of all the interesting parameters you measure from frame to frame. Here's an example:

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
You can create an instance of the Line() class for the left and right lane lines to keep track of recent detections and to perform sanity checks.

Sanity Check
Ok, so your algorithm found some lines. Before moving on, you should check that the detection makes sense. To confirm that your detected lane lines are real, you might consider:

Checking that they have similar curvature
Checking that they are separated by approximately the right distance horizontally
Checking that they are roughly parallel
Look-Ahead Filter
Once you've found the lane lines in one frame of video, and you are reasonably confident they are actually the lines you are looking for, you don't need to search blindly in the next frame. You can simply search within a window around the previous detection.

For example, if you fit a polynomial, then for each y position, you have an x position that represents the lane center from the last frame. Search for the new line within +/- some margin around the old line center.

Double check the bottom of the page here to remind yourself how this works.

Then check that your new line detections makes sense (i.e. expected curvature, separation, and slope).

Reset
If your sanity checks reveal that the lane lines you've detected are problematic for some reason, you can simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for several frames in a row, you should probably start searching from scratch using a histogram and sliding window, or another method, to re-establish your measurement.

Smoothing
Even when everything is working, your line detections will jump around from frame to frame a bit and it can be preferable to smooth over the last n frames of video to obtain a cleaner result. Each time you get a new high-confidence measurement, you can append it to the list of recent measurements and then take an average over n past measurements to obtain the lane position you want to draw onto the image.

Drawing
Once you have a good measurement of the line positions in warped space, it's time to project your measurement back down onto the road! Let's suppose, as in the previous example, you have a warped binary image called warped, and you have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines. You can then project those lines onto the original image as follows:

# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)




