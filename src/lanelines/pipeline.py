import numpy as np
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from lanelines.preprocessing import DistortionCorrector, init_birdeye, image2birdeye, birdeye2image, threshold_binary
from lanelines.detection import fit_polynomial, measure_curvature_pixels, measure_offset_pixels
from lanelines.decorator import weighted_img, image_in_image, text_on_image

from lanelines.io import _raw_video_files

def transform_img(_img, dist_cor, _config, lane_history=None):

    thumbnails = []
    img = np.copy(_img)
    # Undistortion
    img = dist_cor.transform(img)

    # Binary thresholded image
    img = threshold_binary(img)
    thumbnails.append(np.copy(img))

    # Perspective
    img = image2birdeye(img, _config)
    thumbnails.append(np.copy(img))

    # Lane line detection
    img, lane_history = fit_polynomial(img, lane_history)
    thumbnails.append(np.copy(img))

    # Offset and Curvature
    curverad = measure_curvature_pixels(_config.h, lane_history)
    offset = measure_offset_pixels(lane_history, _config)

    # Unwarp
    img = birdeye2image(img, _config)

    # Overlay
    img4 = weighted_img(_img, img, α=0.5, β=1., γ=0.)

    result = image_in_image(img4, thumbnails)
    result = text_on_image(result, curverad, offset)
    return result, lane_history

class StatefulTransformImg(object):
    def __init__(self):
        self.lane_history = None

    def transform_img(self, _img, dist_cor, _config):
        result, lane_history = transform_img(_img, dist_cor, _config, lane_history=self.lane_history)
        self.lane_history = lane_history
        return result

def transform_vid(filename):
    target = _raw_video_files(filename)
    assert len(target) == 1
    print(target)

    # Initialization
    dist_cor = DistortionCorrector()
    calib = dist_cor.fit()

    _config = init_birdeye()

    # Transformation
    on_street_clip = VideoFileClip(target[0])
    img_height, img_width = on_street_clip.size

    stateful = StatefulTransformImg()
    my_func = lambda img: stateful.transform_img(img, dist_cor, _config)

    white_clip = on_street_clip.fl_image(my_func)

    # Close the raw video
    on_street_clip.reader.close()
    on_street_clip.audio.reader.close_proc()

    # Save the processed output
    currDir = os.path.realpath(".")
    rootDir = os.path.abspath(os.path.join(currDir, ".."))
    target_file = os.path.join(rootDir, "data/processed/video_output/") + filename
    white_clip.write_videofile(target_file, audio=False)