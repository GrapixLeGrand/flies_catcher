import cv2 as cv
import numpy as np
import argparse


### start of parsing time

parser = argparse.ArgumentParser(description='Select behavior')
parser.add_argument('--c', help='continous mode, automatically using the cam', action='store_true')
parser.add_argument('--cam', help='if flag set we try to use the cam', action='store_true')

args, _ = parser.parse_known_args()

continous_mode = args.c
use_cam = args.cam

if (continous_mode):
    use_cam = True
    print("--> continous mode enabled")
else:
    print("--> continous mode disabled")

if (use_cam):
    video_stream = cv.VideoCapture(-1)
    print("--> using cam")
else:
    video_stream = None
    print("--> not using cam")

### end of parsing time

#unused for now
MIN_THRESHOLD = 0
MAX_THRESHOLD = 0

"""return the current rgb image, if not using cam, will return always the same image"""
def get_image_rgb():
    if (use_cam):
        was_captured, frame = video_stream.read()
        assert(was_captured)
        return frame
    else:
        return cv.imread(default_target)

"""return the current rgb image, if not using cam, will return always the same image"""
def get_image_grayscale():
    if (use_cam):
        was_captured, frame = video_stream.read()
        assert(was_captured)
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        return cv.imread(default_target, 0)


"""resize the images so that they are in the target"""
def resize_for_target(img, target_dim):
    return cv.resize(img, target_dim, interpolation = cv.INTER_AREA)

def resize_with_factor(img, factor):
    dim = (img.shape[1], img.shape[0])
    dim = (int(dim[0] * factor), int(dim[1] * factor))
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)
    
"""Find the pattern in the target and returns the top-left and bottom right corners if found"""
def find_box_with_pattern(target_img, pattern_img):
    pattern_img_cpy = pattern_img.copy()
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    meth = methods[0]
    w, h = pattern_img.shape[::-1]
    
    method = eval(meth)
    current_target = target_img.copy()
    result = cv.matchTemplate(current_target, pattern_img_cpy, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
            
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print(str(min_val) + " " + str(max_val))
    
    if (max_val < 7000000):
        return None
    
    return (top_left, bottom_right)


"""find a set of corners matching the patterns in the target"""
def find_for_all_patterns(target_img, patterns):
    corners = []
    for p in patterns:
        corners.append(find_box_with_pattern(target_img, p))
    return corners

def find_for_all_patterns_resize(target_img, patterns):
    corners = []
    factors = [1.5, 1, 0.5]
    for p in patterns:
        for f in factors:
            resized_pattern = resize_with_factor(p, f)
            resized_dim = (resized_pattern.shape[1], resized_pattern.shape[0])
            target_dim = (target_img.shape[1], target_img.shape[0])
            if (resized_dim[0] < target_dim[0] and resized_dim[1] < target_dim[1]):
                corner = find_box_with_pattern(target_img, resized_pattern)
                if (corner is not None):
                    corners.append(corner)
    return corners    

def show_labeled_image_with_corners(target_img, corners):
    
    showed_img = target_img.copy()
    
    for c in corners:
        cv.rectangle(showed_img, c[0], c[1], (0, 0, 255), 2)
        
    cv.imshow('image', showed_img)
        

default_target = 'flies2.jpg'
patterns_names = ['fly.jpg', 'fly1.png', 'fly2.png', 'fly3.png', 'fly4.png']
patterns_path = ['pattern/' + p for p in patterns_names]

dummy_image = get_image_rgb() #just to get the initial size
target_dim = (dummy_image.shape[1], dummy_image.shape[0])
fly_dim_pixel = (target_dim[0] // 7, target_dim[1] // 5)

#print(target_dim)
#print(fly_dim_pixel)

patterns = [cv.imread(p,0) for p in patterns_path]
patterns = [resize_for_target(p, fly_dim_pixel) for p in patterns]

exit_requested = False

while(not exit_requested):

    target_grayscale = get_image_grayscale()
    target_rgb = get_image_rgb()
    corners = find_for_all_patterns_resize(target_grayscale, patterns)
    show_labeled_image_with_corners(target_rgb, corners)
    #print(len(corners))
    key = cv.waitKey(1) & 0xFF
    if (key == ord('q') or not continous_mode):
        exit_requested = True

if (not exit_requested):
    cv.waitKey(0)
    
cv.destroyAllWindows() 

print("--> program ended")

