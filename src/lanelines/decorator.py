import numpy as np
import cv2

class ImageDecorator(object):

    def __init__(self):
        pass

    def fit(self):
        return

    def transform(self, _img, _lane_lines, _curvature):
        return np.copy(_img)

def trapezoid(img, src):
    src = np.int32(src)
    img = np.copy(img)
    src = src.reshape((-1,1,2))
    img = cv2.polylines(img, [src],True,(0,255,255), 5)
    return img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def image_in_image(original_image, img_array):

    output_image = np.copy(original_image)
    h, w = original_image.shape[0:2]
    margin = 50

    for pic_indx, pic in enumerate(img_array):
        resized = cv2.resize(pic, (int(w*0.3), int(h*0.3)), interpolation = cv2.INTER_AREA)
        r_h, r_w = resized.shape[0:2]
        
        # Scale up binary thresholded images
        if np.max(resized) == 1:
            resized = resized * 255

        # Add color channels to one channel images
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        output_image[0:r_h, pic_indx * (r_w + margin) : pic_indx * (r_w + margin) + r_w,:] = resized
    return output_image

def text_on_image(original_image, curvature="", offset=""):
    h, w = original_image.shape[0:2]

    text = "Curvature: {0:4.1f} m  Offset: {1:2.2f} m".format(curvature, offset)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_image, text, (10, int(h*0.4)), font, 1, (255,255,255), 2, cv2.LINE_AA)
    return original_image

