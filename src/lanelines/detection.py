import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import cv2

from lanelines.decorator import weighted_img

#def init_lane_lines(_img, _roi_width=800, _min_lane_distance=200):
def init_lane_lines(_img, _roi_width=800, _min_lane_distance=280):
    _img = np.copy(_img)
    # Find peaks in the histogram with min. lane width distance
    _h, _w = _img.shape

    center = int(_w/2)
    _roi_height = int(_h/2)
    roi_x= [int(center - _roi_width/2), int(center + _roi_width/2)]

    roi_y = [_img.shape[0] - _roi_height, _img.shape[0]]

    window_hist = np.sum(_img[roi_y[0] : roi_y[1], roi_x[0] : roi_x[1]], axis=0)
    peaks = find_peaks(window_hist, distance=_min_lane_distance)

    if len(peaks) == 0:
        return None, None

    roi_center = (roi_x[1] - roi_x[0])/2.

    # the left peak is left of the center
    left_lane_peak_candidates = list(filter(lambda p: p < roi_center, peaks[0]))
    right_lane_peak_candidates = list(filter(lambda p: p > roi_center, peaks[0]))

    # Enforce a maximum distance between peaks
    max_dist = 500
    left_lane_peak_candidates2 = list(filter(lambda c: np.any(np.array(right_lane_peak_candidates)-c < max_dist), left_lane_peak_candidates))
    right_lane_peak_candidates2 = list(filter(lambda c: np.any(c-np.array(right_lane_peak_candidates) < max_dist), right_lane_peak_candidates))

    if len(left_lane_peak_candidates2) == 0:
        left_lane_peak = int(center - 200) # Fallback initialization
    else:
        left_lane_peak = max(left_lane_peak_candidates2) + roi_x[0]
        cv2.line(_img,(left_lane_peak,_h),(left_lane_peak,0),1,5)


    if len(right_lane_peak_candidates2) == 0:
        right_lane_peak = int(center + 200) # Fallback initialization
    else:
        right_lane_peak = min(right_lane_peak_candidates2) + roi_x[0]
        cv2.line(_img,(right_lane_peak,_h),(right_lane_peak,0),1,5)

    # right peak is right of the center
    return left_lane_peak, right_lane_peak, _img

def search_sliding_window(binary_warped, lane_history= None):

    leftx_base, rightx_base, _ = init_lane_lines(binary_warped)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped*255, binary_warped*255, binary_warped*255))

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50

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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(255,255,0), 6) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(255,255,0), 6) 
        
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

        # Keep a reasonable distance between the boxes by averaging
        #lane_width = 350
        #leftx_current2 = int((leftx_current + (rightx_current - leftx_current)/2) - lane_width/2)
        #rightx_current2 = int((leftx_current + (rightx_current - leftx_current)/2) + lane_width/2)
        #leftx_current = leftx_current2
        #rightx_current = rightx_current2

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def find_lane_pixels(binary_warped, lane_history=None):

    if (not lane_history is None) and (lane_history.fit_found is True):
        minpix = 200

        # Plausibility Check
        left_fit = np.median(np.array(list(lane_history.left_fits)), axis= 0)
        right_fit = np.median(np.array(list(lane_history.right_fits)), axis= 0)
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        lane_dist_high = (right_fitx[0] - left_fitx[0])
        lane_dist_low = (right_fitx[-1] - left_fitx[-1])
        
        leftx, lefty, rightx, righty, out_img = search_around_poly(binary_warped, lane_history)

        if(len(leftx) >= minpix and len(rightx) >= minpix and 200 < lane_dist_high <  600 and 200 < lane_dist_low < 600):
            return leftx, lefty, rightx, righty, out_img
        
    # No valid poly fit available or possible. Do sliding window.
    print("RESETTING SEARCH")
    return search_sliding_window(binary_warped, lane_history)


def search_around_poly(binary_warped, lane_history):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # SEARCH WINDOW BASED ON MOVING MEDIAN
    left_fit = np.median(np.array(list(lane_history.left_fits)), axis= 0)
    right_fit = np.median(np.array(list(lane_history.right_fits)), axis= 0)

    # Define the search regions around the old fits
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

    # Fit new polynomials
    #left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    result[lefty, leftx] = [255, 0, 0]
    result[righty, rightx] = [0, 0, 255]
    ## End visualization steps ##
    
    return leftx, lefty, rightx, righty, result

from recordclass import recordclass
from collections import deque
LaneLineHistory = recordclass("LaneLineHistory", ["left_fits", "right_fits", "fit_found", "leftx", "lefty", "rightx", "righty"])
def fit_polynomial(binary_warped, lane_history=None):
    # Find our lane pixels first

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, lane_history)
    final_out_img = np.zeros_like(out_img)
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Smoothen the fit if possible
    if not lane_history is None:
        left = np.vstack((np.array([left_fit]), np.array(list(lane_history.left_fits))))
        right = np.vstack((np.array([right_fit]), np.array(list(lane_history.right_fits))))

        left_fit_avg = np.median(left, axis=0)
        right_fit_avg = np.median(right, axis=0)
    else:
        left_fit_avg = left_fit
        right_fit_avg = right_fit


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_found = False
    try:
        left_fitx = left_fit_avg [0]*ploty**2 + left_fit_avg [1]*ploty + left_fit_avg [2]
        right_fitx = right_fit_avg [0]*ploty**2 + right_fit_avg [1]*ploty + right_fit_avg [2]
        fit_found = True
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty # Use the moving average instead!
        right_fitx = 1*ploty**2 + 1*ploty

    # Plots the left and right polynomials on the lane lines
    pts = np.array([list(zip(left_fitx, ploty)) + list(reversed(list(zip(right_fitx, ploty))))],dtype='int32')
    cv2.fillPoly(final_out_img, pts, (0,255,0))

    final_out_img = weighted_img(final_out_img, out_img, α=0.8, β=.3, γ=2.5)

    ## Visualization ##
    # Colors in the left and right lane regions
    final_out_img[lefty, leftx] = [255, 0, 0]
    final_out_img[righty, rightx] = [0, 0, 255]

    # Create an history object for moving average of the fits
    if lane_history is None:
        lane_history = LaneLineHistory(deque([left_fit], maxlen=20), deque([right_fit], maxlen=20), fit_found = fit_found, leftx=leftx, lefty=lefty, rightx=rightx, righty=righty)
    else:
        lane_history.left_fits.append(left_fit)
        lane_history.right_fits.append(right_fit)
        lane_history.fit_found = True
        lane_history.leftx = leftx
        lane_history.lefty = lefty
        lane_history.rightx = rightx
        lane_history.righty = righty

    return final_out_img, lane_history

def measure_curvature_pixels(_h, lane_history):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    x_meter_per_pxl = 0.00986667
    y_meter_per_pxl = 0.0375

    # Transfer pixel coordinates to meters
    leftx = lane_history.leftx * x_meter_per_pxl
    lefty = lane_history.lefty * y_meter_per_pxl
    rightx = lane_history.rightx * x_meter_per_pxl
    righty = lane_history.righty * y_meter_per_pxl

    # Fit a polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, _h-1, _h )
    y_eval = np.max(ploty) * y_meter_per_pxl

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return np.mean([left_curverad, right_curverad])

def measure_offset_pixels(lane_history, _config):
    # Center
    car_position = int(_config.w/2) 

    # Current position
    left_fit = lane_history.left_fits[-1]
    right_fit = lane_history.right_fits[-1]

    left_lane_start = left_fit[0] * (_config.h -1)**2 + left_fit[1] * (_config.h-1) + left_fit[2]
    right_lane_start = right_fit[0] * (_config.h -1)**2 + right_fit[1] * (_config.h-1) + right_fit[2]

    lane_center = int(left_lane_start + (right_lane_start - left_lane_start)/2)

    # Offset
    offset =  car_position - lane_center

    # In meters
    x_meter_per_pxl = 0.00986667
    offset = offset * x_meter_per_pxl

    return offset