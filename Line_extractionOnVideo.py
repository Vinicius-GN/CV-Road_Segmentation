import tensorflow as tf 
import cv2 
import numpy as np 
import RoadLine_Extraction as th

#In the following code, we are using OpenCV`s functions to analise a video and extract the road lines in a specif area we determine
video = cv2.VideoCapture('Images/road_car_view.mp4')

#Function used to extract the raw lines
def Canny(image):
    #grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    HSV_th = th.HSV_threshold(image)
    frame = cv2.GaussianBlur(HSV_th, (5,5), 0) #Reduces the noise if the image
    edges = cv2.Canny(frame, 50, 150)

    return edges

#Creating a dinamic maks to detect the road
def Create_mask(image):
    format = image.shape[0]
    top_triangle_x = 550    
    top_triangle_y = 300

    tmask_points = np.array([(0, 720), (1200, format), (top_triangle_x, top_triangle_y)])

    #Create a mask
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, [tmask_points], (255, 255, 255))
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def cordinates(image, lane):
    slope, intersec = lane
    y1 = image.shape[0]
    y2 = int(y1*(3/4))
    x1 = int((y1 - intersec)/slope)
    x2 = int((y2 - intersec)/slope)
    return np.array([x1,y1,x2,y2])

def unique_line(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2, y2 = line.reshape(4)
            param = np.polyfit((x1,x2), (y1, y2), 1) #Polyfit returns an array of [(curvatura ou derivada), interesecção dos pontos]
            slope = param[0]
            intersection = param[1]
            if slope < 0:#How the pixels values starts from the top left, a slope < 0 represents a left line
                left_lines.append((slope, intersection))
            if slope > 0:#How the pixels values starts from the top left, a slope > 0 represents a right line
                right_lines.append((slope, intersection))
    
    left_average = np.average(left_lines, axis=0) if left_lines else None
    right_average = np.average(right_lines, axis=0) if right_lines else None

    # Check if left or right lines were detected
    if left_average is not None or right_average is not None:
        if left_average is not None and right_average is not None:
            left_line = cordinates(image, left_average)                
            right_line = cordinates(image, right_average)
            return np.array([left_line, right_line])
        elif left_average is not None:
            left_line = cordinates(image, left_average)                
            return np.array([left_line])
        else:
            right_line = cordinates(image, right_average)
            return np.array([right_line])
    else:
        return None
    
 #take the avarege of the lines
def draw_lines(image, Lines):
    road = np.zeros_like(image)
    if Lines is not None:
        for line in Lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(road, (x1,y1), (x2, y2), color=(255, 0, 0), thickness=3)
            cv2.line(image, (x1,y1), (x2, y2), color=(255, 0, 0), thickness=3)
    return road


success, img = video.read()

while success:
    img_lane = np.copy(img)
    canny = Canny(img_lane)
    edges = Create_mask(canny)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    road = draw_lines(img_lane, unique_line(img_lane, lines))
    result = cv2.addWeighted(img_lane, 0.8, road, 1, 1)
    
    #Showing results
    cv2.imshow("Frames", result)
    cv2.imshow("Road", edges)
    cv2.waitKey(1)

    #Now, what if we want to use this code for another road?
    #-For that, there are two different aproaches: create a TF FCN using VGG-16 model or redefine the "top_triangle" variables

    success, img = video.read()

