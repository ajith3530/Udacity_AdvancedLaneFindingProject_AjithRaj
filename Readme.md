## Writeup / README -- Advanced Lane Finding Project -- Ajith Raj

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Output_Images/ALF_CalibratedImage.png "Camera Calibration"
[image2]: ./Output_Images/ALF_UndistortedImage.png "Undistorted"
[image3]: ./Output_Images/ALF_ImageTransformation.png "Road Transformed"
[image4]: ./Output_Images/ALF_BirdsEyeImage.png "Bird's Eye Image"
[image5]: ./Output_Images/ALF_SlidingWindowOutput.png "Fit Visual"
[image6]: ./Output_Images/ALF_FinalResultImage.png "Final Output Image"
[video1]: ./project_video_output.mp4 "Final Output Video"

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

Congratulations, you're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Object points are 3D points in real world space, and Image points are the 2D points in an imput image. For creating Object and Image points for 9x6 chessboard imagesmodifications were made in Section 1 and 2. 
For calculating the distortion corfficients (dist, mtx) and openCV library function *cv2.calibrateCamera()* is used as described in Section 3.
For undistorting an image using the calculated distortion coefficients another openCV library function *cv2.undistort()* was used as described in Section 4.

**Sections of Calibration_function()**
```python
#Section 1 -- Initialization of object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Section 2 -- Running Detection in the Main Loop
ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
if ret == True:
  objpoints.append(objp)
  imgpoints.append(corners)

#Section 3 -- Calculating Distortion Coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

#Section 4 -- Undistorting an Input image based on Distortion Coefficients
UndistortedImage   = cv2.undistort(main_image, self.mtx, self.dist, None, self.mtx) 
```
---

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

An illustration of an undistorted chessboard image is added below.

![alt text][image1]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A color space based binary transformation function is used over Canny or Sobel due to the high robustness in detection accuracy
under stress conditions like low lighting, noise ratio, etc. The LAB color is space is more robust in detecting the white lane, and the RGB 
color space is more robust in detecting the yellow lanes, so a transformation function which leverages both these unique features is used in the section below.
**Section of ImageTransformation_function()**
```python
    ############ LAB Image Processing -- Yellow Lanes
    lab_image    = cv2.cvtColor(main_image , cv2.COLOR_RGB2LAB)
    L,A,B=cv2.split(lab_image)
    B_image = B
    ret,whitelane_image = cv2.threshold(B_image,160,200,cv2.THRESH_BINARY)
   
    ############# RGB Image Processing -- White Lanes   
    lower = np.array([225,180,200],dtype = "uint8")
    upper = np.array([255, 255, 255],dtype = "uint8")
    mask = cv2.inRange(main_image , lower, upper)
    S = cv2.bitwise_and(main_image, main_image, mask = mask)
    S = cv2.cvtColor(S, cv2.COLOR_RGB2GRAY)
    yellowlane_image = S
    ret,S_binary = cv2.threshold(yellowlane_image,160,200,cv2.THRESH_BINARY)
    
    ############ Combining HLS and LAB outputs
    combined_image =  cv2.bitwise_or(yellowlane_image,whitelane_image)
```
An illustration of an binary thresholded image is added below.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The Perspective transform takes the input image and creates a bird's eye view of the image with only  main object of interest in the scope. The area of src and dst points are manually calculated based on hit and trial methods, then openCV library function *cv2.getPerspectiveTransform()* is used to get the perspective Matrix for transformation. (Inverse perspective matrix is also calculated using the same function for unwarping the image in the final steps). And the Bird's eye is generated with the openCV library function *cv2.warpPerspective()* using the perspective Matrix calculated in the previous step.

**Section of PerspectiveTransform_function()**
```python
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    
    leftupperpoint  = [568,470]
    rightupperpoint = [717,470]
    leftlowerpoint  = [260,680]
    rightlowerpoint = [1043,680]

    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
```
An illustration of a Bird's Eye image is added below.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Detection of Lane-line pixels is accomplished using two functions - *Directsearch_function()* and *Indirectsearch_function()*.

**Directsearch_function()**
Directly searches for the Lane markings without considering the previous observations, and returns the location of the
lane markings on the current image. This is a more calculation intensive function.
```python
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 80 

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low       = binary_warped.shape[0] - (window+1)*window_height
        win_y_high      = binary_warped.shape[0] - window*window_height

        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("Value Error Received")
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
```
**Indirectsearch_function()**
Based on the previous observations, searches for the lane markings near the previous lane points. This function is less
calculation intensive.
```python
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  
```
An illustration of a Detected Lane Marking is added below.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and vehicle offset and the offset from the road is calculated using *radius_curvature()* as described below.

```python
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in  space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    # Lane center as mid of left and right lane bottom                        
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix #Convert to meters
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {}".format(center, position)
```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The Entire Pipeline is implemented using a callable class design - *DetectionPipeline*, in which the camera calibration function is called once during the entire execution and rest of the helper functions are called everytime during the entire execution of every frame.

An illustration of a Final Output of the pipeline is added below.

![alt text][image6]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Based on the subclip duration, the output of the Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Problems Faced**
1. Using Class variables in functions caused a lot of unwanted bugs.
2. Creating a callable class with proper initialization was a major pain.
3. Not knowing difference between *cv2.imread()* and *mpimg.imread()*
4. Not converting BGR to RGB, caused the *ImageTransformation_function()* to fail. Debugging this was excruciating.
5. Testing the pipeline with Sobel Operator, faced difficult in removing noise, switched to Color spaces to increase robustness.
6. Not using Error handling while using np.polyfit. (One of the biggest mistakes)

**Failure Regions**
1. Terrain Regions where the lane marking go beyond the manually calculated region coordincates.
2. Detection of lighting conditions for an image, and adjusting threholds for the input image based on the lighting conditions.

**Improvements**
1. Completely remove usage of manual hit and trial methods for finding parameters, develop an algorithm which automatically detects the parameters based on the input image.
2. Add more Test Cases to SanityCheck_function().

 
