import cv2 as cv
import numpy as np

video_stream = cv.VideoCapture(0)

was_captured, frame = video_stream.read()

target = 'flies2.jpg'
patterns = ['fly.jpg', 'fly1.png', 'fly2.png', 'fly3.png', 'fly4.png']
patterns = ['pattern/' + p for p in patterns]

if (was_captured):
    target_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    target_img_colored = frame
else:
    print("failed to read camera")
    target_img = cv.imread(target, 0)
    target_img_colored = cv.imread(target)

target_dim = (target_img.shape[1], target_img.shape[0])
fly_dim_pixel = (target_dim[0] // 7, target_dim[1] // 5)

print(target_dim)
print(fly_dim_pixel)

patterns_imgs = [ cv.imread(p,0) for p in patterns]

"""resize the images so that they are in the target"""
def resize_for_target(img, target_dim):
    return cv.resize(img, target_dim, interpolation = cv.INTER_AREA)

def resize_with_factor(img, factor):
    dim = (img.shape[1], img.shape[0])
    dim = (int(dim[0] * factor), int(dim[1] * factor))
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

patterns_imgs = [resize_for_target(p, fly_dim_pixel) for p in patterns_imgs]
    
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
    cv.waitKey(0)
    cv.destroyAllWindows()     

corners = find_for_all_patterns_resize(target_img, patterns_imgs)
show_labeled_image_with_corners(target_img_colored, corners)
print(len(corners))



