"""
@file : detection.py
@author : Charan Karthikeyan P V
@License : MIT License
@date :11/8/2020
@brief : This file calibrates the camera and gets its calibrated values.
"""
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
%matplotlib inline
"""
@brief : This function imports a chessboard image for reference and
finds the edges and plots them in different colors
@param :
@return :
"""
def edge_points():
	objpoints = []
	imgpoints = []
	images = glob.glob("camera_cal/calibration*.jpg")
	fig_count = len(images)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	for fname in images:
    	img = mpimg.imread(fname)
	    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	    ret, corners = cv2.findChessboardCorners(gray,(9,6), None)
	    
	    if ret == True:
	        objpoints.append(objp)
	        imgpoints.append(corners)
	        
	        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
	        plt.imshow(img)
	        plt.figure()
	 """
	 Uncomment these lines to save the figure and show the output.
	 """       
	# plt.savefig("/home/charan/Desktop/Car/Term1/CarND-Advanced-Lane-Lines/output_images/calibrated_images.jpg")
	# plt.show()

"""
@brief: Function to get the parametes of the camera.
@param : None
@return : None
"""
def calibration():
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


	img = cv2.imread("camera_cal/calibration1.jpg")
	img_size = (img.shape[1], img.shape[0])

	dst  = cv2.undistort(img, mtx, dist, None, mtx)
	cv2.imwrite("camera_cal/test_undist.jpg", dst)

	"""
	Uncomment the below lines to show and save the images and their 
	corresponding calibration params. 
	"""
	# print(ret)
	# print(mtx)
	# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
	# ax1.imshow(img)
	# ax1.set_title('Original Image', fontsize=30)
	# ax2.imshow(dst)
	# ax2.set_title('Undistorted Image', fontsize=30)
	# plt.savefig("camera calibrated image", dst)
	# plt.savefig("/home/charan/Desktop/Car/Term1/CarND-Advanced-Lane-Lines/output_images/calibrated_image.jpg")
	
	"""
	Dump all the calibration parameters for camera in a file.
	"""
	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump(dist_pickle, open("camera_cal/test_dist_pickle.p","wb"))

"""
@brief : Function to read the dumped values.
@param : None.
@return : None.
"""
def readPickle():
	with open("camera_cal/test_dist_pickle.p", "rb") as f:
    data = pickle.load(f)
    print(data["mtx"])
    print(data["dist"])