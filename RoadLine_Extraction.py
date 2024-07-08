import cv2
import numpy as np
import matplotlib.pyplot as plt
import morfological_op as morf

def HSV_threshold(image): 
    img = image  

    #Mean shift test segmentation
    #COLOR_MIN = np.array([18, 0, 208],np.uint8)
    #COLOR_MAX = np.array([60, 76, 222],np.uint8)

    #Segmentation2
    #COLOR_MIN = np.array([163, 247, 224],np.uint8)
    #COLOR_MAX = np.array([176, 255, 235],np.uint8)

    #Segmentation4
    #COLOR_MIN = np.array([148, 0, 193],np.uint8)
    #COLOR_MAX = np.array([152, 233, 201],np.uint8)

    #SP0 (shadows)
    #COLOR_MIN = np.array([98, 89, 124],np.uint8)
    #COLOR_MAX = np.array([171, 140, 220],np.uint8)

    #BP (Bright and painted road)
    #COLOR_MIN = np.array([0, 0, 209],np.uint8)
    #COLOR_MAX = np.array([65, 57, 255],np.uint8)

    #BP6-> different approach
    COLOR_MIN = np.array([5,0, 60],np.uint8)
    COLOR_MAX = np.array([179, 35, 153],np.uint8)

    #SP4 -> different approach
    #COLOR_MIN = np.array([88,0, 62],np.uint8)
    #COLOR_MAX = np.array([179, 92, 126],np.uint8)

    #SP4
    #COLOR_MIN = np.array([80, 17, 144],np.uint8)
    #COLOR_MAX = np.array([161, 87, 249],np.uint8)

    #Mean shift test segmentation
    #COLOR_MIN = np.array([18, 0, 208],np.uint8)
    #COLOR_MAX = np.array([60, 76, 222],np.uint8)

    #Mean shift test segmentation1
    #COLOR_MIN = np.array([40, 0, 173],np.uint8)
    #COLOR_MAX = np.array([47, 86, 237],np.uint8)

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    return frame_threshed

def morf_operations(image, kernel):
    HSV_th = HSV_threshold(image)
    cv2.imshow("HSV", HSV_th)
    cv2.waitKey(0)

    #BP6 -> different approach
    opening_img = morf.closing(HSV_th, kernel)
    cv2.imshow("OP1", opening_img)
    cv2.waitKey(0)

    opening_img1 = morf.closing(opening_img , kernel)
    cv2.imshow("OP2", opening_img1)
    cv2.waitKey(0)

    opening_img1 = morf.opening(opening_img , kernel)
    cv2.imshow("OP3", opening_img1)
    cv2.waitKey(0)

    #kmean_seg     
    """erode_img = morf.erode(HSV_th, kernel)
    cv2.imshow("OP1", erode_img)
    cv2.waitKey(0)

    closing_img = morf.closing(erode_img, kernel)
    cv2.imshow("OP2", closing_img)
    cv2.waitKey(0)

    erode_img = morf.erode(closing_img, kernel)
    cv2.imshow("OP3", erode_img)
    cv2.waitKey(0)

    closing_img = morf.closing(erode_img, kernel)
    cv2.imshow("OP4", closing_img)
    cv2.waitKey(0)"""

    #Segmentation1       
    """erode_img = morf.erode(HSV_th, kernel)
    cv2.imshow("OP1", erode_img)
    cv2.waitKey(0)"""

    #SP4 -> different aproach
    """opening_img = morf.opening(closing_img, kernel)
    cv2.imshow("OP2", opening_img)
    cv2.waitKey(0)

    opening_img1 = morf.opening (opening_img, kernel)
    cv2.imshow("OP3", opening_img1)
    cv2.waitKey(0)"""

    #SP4
    """dilate_img = morf.dilate(opening_img, kernel)
    cv2.imshow("Dilate", dilate_img)
    cv2.waitKey(0)"""

    #SP0
    """kernel1 = morf.create_kernel(2, 'ellipse')

    dilate_img = morf.dilate(HSV_th, kernel1)
    cv2.imshow("Dilate0", dilate_img)
    cv2.waitKey(0)

    dilate_img1 = morf.dilate(dilate_img, kernel1)
    cv2.imshow("Dilate1", dilate_img1)
    cv2.waitKey(0)

    erode_img = morf.erode(dilate_img1, kernel)
    cv2.imshow("Erode0", erode_img)
    cv2.waitKey(0)

    closing_img = morf.closing(erode_img, kernel)
    cv2.imshow("Closing", closing_img)
    cv2.waitKey(0)"""
    
    #BP6

    """opening_img2 = morf.opening(opening_img, kernel)
    cv2.imshow("OP2", opening_img2)
    cv2.waitKey(0)

    kernel1 = morf.create_kernel(3, 'cross')

    opening_img3 = morf.opening(opening_img2, kernel1)
    cv2.imshow("OP3", opening_img3)
    cv2.waitKey(0)
     
    erode_img = morf.erode(opening_img3, kernel1)
    cv2.imshow("cL3", erode_img)
    cv2.waitKey(0)

    kernel2 = morf.create_kernel(2, 'cross')

    closing_img = morf.closing(opening_img3, kernel2)
    cv2.imshow("OP4", closing_img)
    cv2.waitKey(0)"""

    return opening_img1

def Canny(image):
    frame = cv2.GaussianBlur(image, (5,5), 0) #Reduces the noise 

    cv2.imshow("gaussian_b", frame)
    cv2.waitKey(0)

    edges = cv2.Canny(frame, 50, 200, None, 3)

    cv2.imshow("Canny", edges)
    cv2.waitKey(0)
    return edges

def Create_mask(image):
    w = image.shape[0]
    h = image.shape[1]
    top_rec_x, top_rec_y = 0, 205
    top_rec_x2, top_rec_y2 = h, 205

    print("Width: ", w)
    print("Height: ", h)
    tmask_points = np.array([(top_rec_x, top_rec_y), (top_rec_x2, top_rec_y2), (h, w), (0, w)])

    #Create a mask
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, [tmask_points], (255, 255, 255))
    masked_img = cv2.bitwise_and(image, mask)
    cv2.imshow("Mask", masked_img)
    cv2.waitKey(0)
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
            print(line)
            param = np.polyfit((x1,x2), (y1, y2), 1) #Polyfit returns an array of [(slope), interesection]
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
            print("Both sides")
            return np.array([left_line, right_line])
        elif left_average is not None:
            left_line = cordinates(image, left_average)       
            print("Left side")         
            return np.array([left_line])
        else:
            right_line = cordinates(image, right_average)
            print("Right side") 
            return np.array([right_line])
    else:
        return None
    
 #take the avarege of the lines
def draw_lines(image, Lines):
    road = np.zeros_like(image)
    if Lines is not None:
        for line in Lines:
            x1,y1,x2,y2 = line.reshape(4)
            if(abs(y1-y2) > 10): #Avoid drawing horizontal lines
                cv2.line(road, (x1,y1), (x2, y2), color=(0, 0, 0), thickness=3)
                cv2.line(image, (x1,y1), (x2, y2), color=(0, 0, 0), thickness=3)
    return road

def main():
    #image_number = str(input("Enter the image code: "))
    img_main = cv2.imread(f'Images/BP6.png')
    cv2.imshow("Img_main", img_main)
    cv2.waitKey(0)

    if img_main is None:
        print("Image not found")
        exit()

    mask_img = Create_mask(img_main)
    img_morf = morf_operations(mask_img, morf.create_kernel(4, 'ellipse'))
    edges_road = Canny(img_morf) 

    lines = cv2.HoughLinesP(edges_road, rho=2, theta=np.pi/180, threshold=100, minLineLength=20, maxLineGap=3)

    road = draw_lines(img_main, lines)

    result = cv2.addWeighted(img_main, 0.8, road, 1, 1)
        
    #Showing results
    cv2.imshow("Frames", result)
    cv2.waitKey(0) 



if __name__ == "__main__":
    main()
