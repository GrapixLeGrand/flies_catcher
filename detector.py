import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse

#####################################################################################
# Constants
#####################################################################################

#allows to run on a computer without cam
DEFAULT_TARGET = cv.imread("flies.jpg")

#opencv output
WINDOW_DIMS = (1500, 200)

#pipeline constants
WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)
DISTANCE_THRESHOLD = 10
GRAYSCALE_THRESHOLD = 120
RESIZE_FACTOR = 50
LOW_PASS_FILTER_KERNEL_DIMS = (4, 4)
BLOBS_SIZES_BOUNDS = (7, 10)

#for the blob detection
BLOB_DETECTOR_PARAMS = cv.SimpleBlobDetector_Params()
BLOB_DETECTOR_PARAMS.filterByConvexity = False
BLOB_DETECTOR_PARAMS.filterByCircularity = False
BLOB_DETECTOR_PARAMS.filterByInertia = False
BLOB_DETECTOR_PARAMS.filterByArea = False

#utils
EXIT_KEY = 'q'
VIDEO_SOURCE_INDEX = -1


#####################################################################################
# Argument
#####################################################################################

parser = argparse.ArgumentParser(description='Select behavior')
parser.add_argument('--continous', help='continous mode, automatically using the cam', action='store_true')
parser.add_argument('--cam', help='if flag set we try to use the cam', action='store_true')
args, _ = parser.parse_known_args()

CONTINOUS_MODE = args.continous
CAM_MODE = args.cam

if (CONTINOUS_MODE):
    CAM_MODE = True
    print("--> continous mode enabled, therefore using the cam")
else:
    print("--> continous mode disabled")

if (CAM_MODE):
    video_stream = cv.VideoCapture(VIDEO_SOURCE_INDEX)
    print("--> using cam")
else:
    video_stream = None
    print("--> not using cam")

"""
return the current rgb image, if not using cam, will return always the same image.
We are adding this level of indirection to allow using this program without a cam.
"""
def get_image_rgb():
    if (CAM_MODE):
        was_captured, frame = video_stream.read()
        assert(was_captured)
        return frame
    else:
        return DEFAULT_TARGET


#####################################################################################
# Detection part
#####################################################################################

"""
this method takes a vector of rgb components @rgb_vec and
replace it with @replace if the average absolute difference
with each component regarding the average of the components
exceeds the threshold. In other word, this function aim to
replace the vector by another if the pixel is not as grayscaled
as we want
"""
def compute_avg_diff_comp(rgb_vec, replace, thresh):
    assert(rgb_vec.shape[0] == 3)
    mean = np.mean(rgb_vec)
    sub = np.abs(rgb_vec - mean)
    sub_mean = np.max(sub)
    if (sub_mean < thresh):
        return rgb_vec
    else:
        return replace
        
"""
this methods takes an image and filter it with @compute_avg_diff_comp
the resulting image is an image where the pixels not satisfying the threshod
are full white.
"""
def filter_threshold_colored_pixels(img):
    global WHITE_RGB
    global DISTANCE_THRESHOLD
    result = []
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            rgb_vec = img[i][j]
            result.append(compute_avg_diff_comp(rgb_vec, WHITE_RGB, DISTANCE_THRESHOLD))
    result = np.asarray(result, dtype=np.uint8)
    result = result.reshape((img.shape[0], img.shape[1], img.shape[2]))
    return result


"""
convert an rgb numpy array to a grayscale numpy array
the type must be uint8 for each component
"""
def convert_rgb_to_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

"""
convert permute all the red and blue channels of an image.
This is needed for opencv
"""
def convert_bgr_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

"""
given a grayscale image, set the pixel having 
brightness greater than threshold to white
"""
def filter_threshold_binary_grayscale_to_white(img):
    global GRAYSCALE_THRESHOLD
    img[img > GRAYSCALE_THRESHOLD] = 255
    return img

"""
set all non white pixels to black
"""
def filter_threshold_binary_non_white_to_black(img):
    img[img != 255] = 0
    return img

"""
this function resize the image by a RESIZE_FACTOR
"""
def resize_img(img):
    global RESIZE_FACTOR
    width = img.shape[1]
    height = img.shape[0]
    max_dim = max(width, height)
    width_adjusted = int((width / max_dim) * RESIZE_FACTOR)
    height_adjusted = int((height / max_dim) * RESIZE_FACTOR)
    dim_adjuted = (width_adjusted, height_adjusted)
    resized = cv.resize(img, dim_adjuted, interpolation = cv.INTER_AREA)
    return resized

"""
This functions apply a low pass filter kernel to the image
"""
def filter_low_pass_img(img):
    global LOW_PASS_FILTER_KERNEL_DIMS
    return cv.blur(img, LOW_PASS_FILTER_KERNEL_DIMS)

"""
find all the blobs in the image and highlight them.
We also return a list of the blob sizes
"""
def find_blobs(img):
    global BLOB_DETECTOR_PARAMS
    detector = cv.SimpleBlobDetector_create(BLOB_DETECTOR_PARAMS)
    keypoints = detector.detect(img)
    blobs_sizes = []
    for k in keypoints:
        blobs_sizes.append(k.size)
    blobs_sizes = np.array(blobs_sizes)
    return cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), blobs_sizes

"""
set all boarders pixels of the image to the value given as argument
"""
def set_image_boarders(img, value):
    assert(value >= 0 and value <= 255)
    w = img.shape[1]
    h = img.shape[0]
    img[0, :, ] = value
    img[h - 1, :, ] = value
    img[:, 0, ] = value
    img[:, w - 1, ] = value
    return img

"""
this function removes any external layer of pixel in the border of the
image. This is helpful for the blob detection part.
"""
def set_boarders_white(img):
    return set_image_boarders(img, 255)


def set_boarders_black(img):
    return set_image_boarders(img, 0)

"""
preproessing pipeline:

1. we resize the image 
    This allows us to remove a bit of noise and to allows more efficient calculations
2. We filter the colored objects to white
    Since a fly is gray to black we want to get rid of all the colored pixels in the image
3. We convert the resulting image to grayscale because now all three components are almost equal
4. We filter pixels by their brightness since a fly is dark
5. We filter all non white pixels to black because each non white pixel at this point is a good
    candidate for belonging to a fly
6. We blur the image to remove high frequency data occuring
7. We filter again all non white pixel to black, this enlight area of interest
"""

pipeline = [
    (resize_img, "resized"),
    (filter_threshold_colored_pixels, "filter colored pixel"),
    (convert_rgb_to_grayscale, "convert to grayscale"),
    (filter_threshold_binary_grayscale_to_white, "threshold brightness"),
    (filter_threshold_binary_non_white_to_black, "set non white to black"),
    (filter_low_pass_img, "low pass filter"),
    (filter_threshold_binary_non_white_to_black, "set non white to black again"),
    (set_boarders_white, "remove external pixel line")
]


"""
pipeline processing the image in order to prepare it to be detected
this function should only be used in a python notebook
"""
def preprocessing_pipeline_matplotlib(img):
    plt.figure()
    f, axarr = plt.subplots(len(pipeline) + 2, 1, figsize=(40, 40)) 
    axarr[0].set_title("orignial input " + str(img.shape[1]) + "x" + str(img.shape[0]))
    axarr[0].imshow(img)
    
    pipeline_result_img = None
    
    for i in range(1, len(pipeline) + 1):
        pipeline_step = pipeline[i - 1]
        func = pipeline_step[0]
        img = func(img)
        
        if (i == len(pipeline) - 1):
            pipeline_result_img = img
        
        if (len(img.shape) == 3):
            assert(img.shape[2] == 3)
            axarr[i].imshow(img)
        else:
            axarr[i].imshow(img, cmap = plt.cm.gray)
        axarr[i].set_title(pipeline_step[1] + " " + str(img.shape[1]) + "x" + str(img.shape[0]))
    
    assert(pipeline_result_img is not None)
    detection_image, blob_sizes = find_blobs(pipeline_result_img)
    axarr[len(pipeline) + 1].imshow(detection_image)
    axarr[len(pipeline) + 1].set_title("found " + str(len(blob_sizes)) + " blobs")

    plt.show()
    
    return img

 
#preprocessed_img = preprocessing_pipeline_matplotlib(convert_bgr_to_rgb(img))


"""returns an image with all the pipeline"""
def pipeline_opencv(img):
    images = []
    pipeline_result_img = None
    
    for i in range(0, len(pipeline)):
        pipeline_step = pipeline[i]
        func = pipeline_step[0]
        img = func(img)
        
        if (i == len(pipeline) - 1):
            pipeline_result_img = img
        
        img_cpy_black_boarders = set_boarders_black(img.copy())
        #if the output is colored or not
        if (len(img.shape) == 3):
            assert(img.shape[2] == 3)
            images.append(img_cpy_black_boarders)
        else:
            images.append(cv.cvtColor(img_cpy_black_boarders, cv.COLOR_GRAY2BGR))
            
    assert(pipeline_result_img is not None)
    detection_image, blob_sizes = find_blobs(pipeline_result_img)
    images.append(set_boarders_black(detection_image))
    
    return blob_sizes, np.concatenate(images, axis=1)


exit_requested = False
error = False

cv.namedWindow('result', cv.WINDOW_NORMAL)
cv.resizeWindow('result', WINDOW_DIMS[0], WINDOW_DIMS[1])

try:
    while (not exit_requested):
        
        img = get_image_rgb()
        starting_time = round(time.time() * 1000)
        blob_sizes, res = pipeline_opencv(img)
        print("t = ", str(round(time.time() * 1000) - starting_time), " ms")

        cv.imshow('result', res)

        #threshold detection

        key = cv.waitKey(1) & 0xFF
        if (key == ord(EXIT_KEY) or not CONTINOUS_MODE):
            exit_requested = True

except:
    #to ensure that we detroy the opencv window on problem
    error = True
    print("program crashed")
finally:
    if (not CONTINOUS_MODE and not error):
        cv.waitKey()
    cv.destroyAllWindows()

print("--> program ended")