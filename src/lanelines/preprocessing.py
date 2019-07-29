from collections import namedtuple
import glob as glob
import cv2
import numpy as np
from lanelines.io import camera_calibration_files

# Define data type for camera calibrations
CameraCalibration = namedtuple("CameraCalibratioon", ["ret", "mtx", "dist", "rvecs", "tvecs"])
class DistortionCorrector(object):
    def __init__(self, _camera_calibration=None):
        # initialize
        self._camera_calibration = None if _camera_calibration is None else _camera_calibration
        self._h, self._w = None, None
        return
    
    """
    Perform camera calibration
    """
    def fit(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = camera_calibration_files()

        # Step through the list and search for chessboard corners
        if len(images) == 0:
            raise Exception("No camera calibration images found.")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Save resolution of images that were initialized
        self._h,  self._w = gray.shape[:2]

        # Create the new calibration object
        self._camera_calibration = CameraCalibration(*cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None))
        print("Successfully calibrated DistortionCorrector.")
        return self._camera_calibration

    def transform(self, _img):
        assert _img.shape[:2] == (self._h, self._w)
        assert not self._camera_calibration is None

        # undistort
        dst = cv2.undistort(_img, self._camera_calibration.mtx, self._camera_calibration.dist, None, self._camera_calibration.mtx)
        return dst

    def get_info(self):
        return self._camera_calibration, self._h, self._w

"""
PERSPECTIVE TRANSFORMATION
"""

PerspectiveTransformationConfig = namedtuple("PerspectiveTransformationConfig", ["M", "w", "h", "src", "dst"])
def init_birdeye(img_h=720, img_w=1280, trpzd_y_start=445, trpzd_height=225, trpzd_width_low=250, trpzd_width_high=2500):
    h, w = img_h, img_w
    center = int(w/2)
    src = np.array([[center-int(trpzd_width_low/2), trpzd_y_start],
                    [center+int(trpzd_width_low/2), trpzd_y_start],
                    [center+int(trpzd_width_high/2), trpzd_y_start + trpzd_height],
                    [center-int(trpzd_width_high/2), trpzd_y_start + trpzd_height]], np.float32)

    dst = np.array([[0, 0],
                    [w, 0],
                    [w, h],
                    [0, h]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return PerspectiveTransformationConfig(M, w, h, src, dst)

def image2birdeye(_img, _config):
    assert _img.shape[0] == _config.h
    assert _img.shape[1] == _config.w
    return cv2.warpPerspective(_img, _config.M, (_config.w, _config.h))

def birdeye2image(_img, _config):
    assert _img.shape[0] == _config.h
    assert _img.shape[1] == _config.w
    return cv2.warpPerspective(_img, _config.M, (_config.w, _config.h), \
            flags=cv2.WARP_INVERSE_MAP)

"""
Convert a given RGB or BGR image to HLS and return each channel seperately.
"""
def threshold_binary(_img):
    # Color thresholding
    _, _ , s = hls(_img)
    s_imgs_thrsh = threshold(s, thresh=[170, 255])

    # Sobel thresholding
    img_gray = cv2.GaussianBlur(gray(_img),(3,3),cv2.BORDER_DEFAULT)

    sobel_x = threshold(scale(Sobel(img_gray, orientation="x", sobel_kernel=3)), thresh=[20, 100])
    sobel_y = threshold(scale(Sobel(img_gray, orientation="y", sobel_kernel=3)), thresh=[20, 100])
    sobel_xy = np.zeros_like(sobel_x)
    sobel_xy[(sobel_x == 1) & (sobel_y == 1)] = 1

    # Combined
    color_binary = np.zeros_like(sobel_xy)
    color_binary[(sobel_xy == 1) | (s_imgs_thrsh == 1)] = 1

    return color_binary

def hls(_img, clr_encoding="RGB"):
    assert clr_encoding in ["RGB", "BGR"]
    if clr_encoding == "RGB":
        h, l, s = cv2.split(cv2.cvtColor(_img, cv2.COLOR_RGB2HLS))
    else:
        h, l, s = cv2.split(cv2.cvtColor(_img, cv2.COLOR_BGR2HLS))
    return h, l, s


def gray( _img, clr_encoding="RGB"):
    """Short summary of the functionality

    Convert a given RGB or BGR image to HLS and return each channel seperately.

    Parameters
    ----------
    _img_channel : array-like, shape = [n_samples]
        A short description

    Returns
    -------
    self : object
    """
    assert clr_encoding in ["RGB", "BGR"]
    if clr_encoding == "RGB":
        param = cv2.COLOR_BGR2GRAY
    else:
        param = cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(_img, param)
    return gray

"""
Creates a scaled Sobel in either x or y axis.
"""
def Sobel(_img_channel, orientation="x", sobel_kernel=3):
    assert orientation in ["x", "y"]
    if orientation == "x":
        abs_sobel = np.absolute(cv2.Sobel(_img_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(_img_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    return abs_sobel


"""
Scales a given channel of an image so that the maximum value is 255.
"""
def scale(_img_channel):
    scaled_img_channel = np.uint8(255*_img_channel/np.max(_img_channel))
    return scaled_img_channel

"""
Applies a threshold on a 1-channel image like a scaled sobel or a single
color channel.
"""
def threshold(_img_channel, thresh=[0, 255]):
    # Threshold gradient
    binary_output = np.zeros_like(_img_channel, dtype="uint8")
    binary_output[(_img_channel >= thresh[0]) & (_img_channel <= thresh[1])] = 1
    return binary_output