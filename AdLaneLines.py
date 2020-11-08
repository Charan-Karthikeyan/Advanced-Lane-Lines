"""
@file : AdLaneLines.py
@author : Charan Karthikeyan P V
@License : MIT License
@date :11/08/2020
@brief : This file is to detect the lane lines on the read and 
give output for the curvature and better detection methods.
"""

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

"""
@brief : Read the dumped data with the camera calibration parameters
@param : pickle_file -> The path of the pickle file
@return : The arrays containing the calibration data.
"""
def calib(pickle_file):
	with open(pickle_file, "rb") as f:
		data = pickle.load(f)
		return data["mtx"], data["dist"]

"""
@brief : Function to read each frame in the video and undistort them 
with the camera calibration parameters.
@param : img -> The input fram from the image.
		 mtx, dist -> The camera calibration paramerters
@return : The distorted image. 
"""
def undist(img, mtx, dist):
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst

"""
@brief : Function to apply Sobel threshold on the image for better
feature extraction.
@param : img -> The input frame.
		 solbel_kernel -> The size of the kernel that the kernel works on.
		 orient -> The orientation of image and the type of threshold to implement.
		 thresh -> The Threshold of the sobel to be applied.
@return : The image with the threshold applied on.
"""
def absSobelThresh(img, sobel_kernel=3, orient='x', thresh=(20,100)):
	threshMin = thresh[0]
	threshMax = thresh[1]
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)
	elif orient =='y':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=sobel_kernel)
	absSobel = np.absolute(sobel)
	scaledSobel = np.uint8(255*absSobel/np.max(absSobel))
	sxbinary = np.zeros_like(scaledSobel)
	sxbinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
	
	return sxbinary
"""
@brief : Function to convert the given image into a binary image.
@param : img -> The input image
		 thresh_s, sx, cs, cx -> The various threshold for the channels in the image.
@return : The final binary image.
"""
def binary_transformation(img,thresh_s=(20,100), thresh_sx=(20,100), thresh_cs=(170,255),thresh_cu=(0,110), thresh_cr=(230,255)):
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	rgb = img.astype(np.float)
	yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float)
	
	r_channel = rgb[:,:,0]
	u_channel = yuv[:,:,1]
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	
	abs_sobelx = absSobelThresh(img, orient='x', thresh=(thresh_s[0], thresh_s[1]))
	abs_sobely = absSobelThresh(img, orient='y', thresh=(thresh_sx[0], thresh_sx[1]))
	
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= thresh_cs[0]) & (s_channel <= thresh_cs[1])] = 1
	
	r_binary = np.zeros_like(r_channel)
	r_binary[(r_channel >= thresh_cr[0]) & (r_channel <= thresh_cr[1])] = 1
	
	u_binary = np.zeros_like(u_channel)
	u_binary[(u_channel >= thresh_cu[0]) & (u_channel <= thresh_cu[1])] = 1
	
	clr_binary = np.dstack((r_binary, ((abs_sobelx == 1) & (abs_sobely == 1)), u_binary))
	
	combined_binary = np.zeros_like(s_binary)
	combined_binary[(r_binary == 1) | (u_binary == 1) | ((abs_sobelx == 1) & (abs_sobely == 1))] = 1
	return combined_binary

"""
@brief : Function to perform perspective transformation.
@param  : img -> the image to warp.
@return :The warped image, transformation matrix and its inverse.
"""
def warp(img):
	img_size = (img.shape[1], img.shape[0])
	src = np.float32([[570, 470],[720,470],[260,680],[1040,680]])
	dst = np.float32([[200,0],[1000,0],[200,700],[1000,700]])
	#offset = 20
	#dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset],[offset, img_size[1]-offset]])
	
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(img , M, img_size, flags=cv2.INTER_LINEAR)
	return warped, M ,Minv

"""
@brief: Funtion to visually depict the placement of the lane lines as a grah
@praam : img -> The image to find the histogram on.
@return : None.
"""
def display_histogram(img):
	hist = np.sum(img[int(img.shape[0]/2):,:], axis=0)
	"""
	Uncomment the below lines to see the graph generated from the image.
	"""
	# plt.title("Histogram")
	# plt.plot(hist)
	return 

"""
@brief : Function that detects the lane lines and gives a 
iterarive bounding boxes and jinted together to form a line 
even when the lane lines are not continuous and helps get the curvature of the
road
@param : binary_warped -> The binary warped image from the video 
@return : The image, curvature radius, offset of vehicle from center, left and 
right lane lines and their plots.
"""
def window_fit(binary_warped):
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 10
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #plt.imshow(out_img)
    #plt.savefig("/home/charan/Desktop/Car/Term1/CarND-Advanced-Lane-Lines/output_images/sliding_window")
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='red')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    #print(left_curverad,'m', right_curverad,'m')
    #curvature = (left_curverad + right_curverad) / 2
    
    m_car = binary_warped.shape[1] / 2
    m_lane = (left_fitx[0] + right_fitx[0]) / 2
    offset_right_from_center_m = (m_lane-m_car)*xm_per_pix
    
    avg_radius_meters = np.mean([left_curverad, right_curverad])
    
    return out_img, avg_radius_meters, offset_right_from_center_m, left_fitx, right_fitx, ploty


"""
@brief : Function to drwa the lane lines on the image.
@param : image -> The original image from the video frame.
		 warped -> The corresponding warped image from the image.
		 Minv -> the inverse matrix from perspective transformation.
		 left, right_fitx, ploty -> The lane lines detected and plotted
		 r_meters -> The radius of the curvature.
@return : Array containing the final image and the image fit.
"""
def draw_line(image, warped, Minv, left_fitx, right_fitx, ploty, r_meters):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (236,201, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    """
    Uncomment below lines to view and save corresponding output
    """
    #plt.imshow(result)
    # plt.savefig("output_images/draw_line")
    #add_figure_to_image(result, curvature=r_meters, vehicle position = right_from_center_m, left Coeff = l_fit, Right Coeff = r_fit )
    return result

"""
The main file to run all the code in this file 
"""
white_output = "P4.mp4"
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)


