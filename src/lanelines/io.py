from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import glob
import os
import cv2

def _raw_image_files(fname="*.jpg"):
    currDir = os.path.realpath(".")
    rootDir = os.path.abspath(os.path.join(currDir, ".."))
    return glob.glob(os.path.join(rootDir, "data/raw/test_images/") + fname)

def camera_calibration_files(fname="calibration*.jpg"):
    currDir = os.path.realpath(".")
    rootDir = os.path.abspath(os.path.join(currDir, ".."))
    return glob.glob(os.path.join(rootDir, "data/raw/camera_cal/") + fname)

def _raw_video_files(fname="*.mp4"):
    currDir = os.path.realpath(".")
    rootDir = os.path.abspath(os.path.join(currDir, ".."))
    return glob.glob(os.path.join(rootDir, "data/raw/test_videos/") + fname)

def load_image(_img_fname):
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
    imgs = _raw_image_files(fname=_img_fname)
    assert len(imgs) == 1
    ret = cv2.imread(imgs[0])
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)
    return ret

def load_frame(_video_fname, subclip=[0, 50], frame_offset=0.0):
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
    # Get path to target video
    vid_file = _raw_video_files(fname=_video_fname)
    assert len(vid_file) == 1

    # Load video and close reader
    on_street_clip = VideoFileClip(vid_file[0]).subclip(subclip[0],subclip[1])
    ret = on_street_clip.get_frame(frame_offset)
    on_street_clip.reader.close()
    on_street_clip.audio.reader.close_proc()

    # Convert to RGB color space
    #ret = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return ret

def plot_images(fnames, col_names,*args):
    fig = plt.figure(figsize=(25,35))
    fig.tight_layout()
    
    n_columns = len(args)
    n_rows = len(args[0])

    pos_count = 1
    for i in range(n_rows):
        # plot the images of a row
        for j in range(n_columns):
            plt.subplot(n_rows, n_columns, pos_count)
            plt.axis("off")
            plt.imshow(args[j][i]) if len(args[j][i].shape) > 2 else plt.imshow(args[j][i], cmap="gray")
            plt.title(fnames[i] + " " + col_names[j], fontsize=16)
            pos_count += 1