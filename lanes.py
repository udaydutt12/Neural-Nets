import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 -intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope,intercept = parameters[0] , parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])

# blur the image to filter out noise (smoothing the image)
# kernel of convolution:
# 1 2 1
# 2 4 2   <-- 3x3 kernel
# 1 2 1
# use Gaussiam blur kernel with a 5x5 kernel, and a std dev of 0
# now do canny edge detection
# find the gradient of change in intensity values (derivative)
# using canny
def canny(img):
    if len(img.shape)==3:
        img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(img,ksize = (5,5),sigmaX = 0)
    canny = cv.Canny(blur,threshold1 = 50, threshold2 = 150)  # doc says use low:high of 1:2 or 1:3
    return canny

def region_of_interest(image):
    height = image.shape[0]
    #triangle = np,array([(200,height),(1100,height),(550,250)])
    polygons = np.array([
        [(200,height),(1100,height),(550,250)]
    ])
    #creates a black mask of size of input image
    mask = np.zeros_like(image)
    #fills the mask with 255 for all polygons specified
    # 255 = white for grayscale
    cv.fillPoly(img = mask, pts = polygons, color = 255)
    return mask & image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv.line(line_image,(x1,y1),(x2,y2),color = (255,0,0), thickness=10)
    return line_image

'''
#img = cv.imread('test_image.jpg', flags = cv.IMREAD_GRAYSCALE)
img = cv.imread('test_image.jpg')
lane_image = np.copy(img)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
lines = cv.HoughLinesP(image = cropped_image, rho = 2 ,theta = np.pi/180, threshold = 100, lines = np.array([]),minLineLength=40,maxLineGap=5)
#line_image = display_lines(image = lane_image,lines = lines)
averaged_lines = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,averaged_lines)
combo_image = cv.addWeighted(src1= lane_image,alpha = 0.8,src2 = line_image,beta = 1 , gamma= 1)
cv.imshow('result',combo_image)
cv.waitKey(0)
#plt.imshow(canny)
#plt.show() # equivalent to waitkey(0)
'''

cap = cv.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(image = cropped_image, rho = 2 ,theta = np.pi/180, threshold = 100, lines = np.array([]),minLineLength=40,maxLineGap=5)
    #line_image = display_lines(image = frame,lines = lines)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv.addWeighted(src1= frame,alpha = 0.8,src2 = line_image,beta = 1 , gamma= 1)
    cv.imshow('result',combo_image)
    if cv.waitKey(1) == ord('q'): # wait 1ms
        break
cap.release()
cv.destroyAllWindows()

