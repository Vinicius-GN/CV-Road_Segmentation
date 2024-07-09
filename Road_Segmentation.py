import cv2
import numpy as np
import matplotlib.pyplot as plt
import morfological_op as morf

# Function to apply threshold on an image using the HSV color space
def HSV_threshold(image): 
    img = image  

    # Here we have the different color ranges for the road segmentation depending on the image
    
    # kmean4
    # COLOR_MIN = np.array([116, 52, 34],np.uint8)
    # COLOR_MAX = np.array([144, 68, 51],np.uint8)
    
    # means4
    # COLOR_MIN = np.array([33, 83, 178],np.uint8)
    # COLOR_MAX = np.array([49, 95, 249],np.uint8)

    # means6
    # COLOR_MIN = np.array([19, 200, 134],np.uint8)
    # COLOR_MAX = np.array([33, 233, 151],np.uint8)
    
    # kmean6
    # COLOR_MIN = np.array([101, 62, 3],np.uint8)
    # COLOR_MAX = np.array([179, 255, 255],np.uint8)

    # means5
    # COLOR_MIN = np.array([172, 92, 121],np.uint8)
    # COLOR_MAX = np.array([179, 120, 135],np.uint8)

    # means_h1
    # COLOR_MIN = np.array([105, 97, 148],np.uint8)
    # COLOR_MAX = np.array([118, 130, 167],np.uint8)  

    # means_h3
    # COLOR_MIN = np.array([53, 177, 131],np.uint8)
    # COLOR_MAX = np.array([59, 198, 160],np.uint8)

    # means_h4
    # COLOR_MIN = np.array([52, 50, 207],np.uint8)
    # COLOR_MAX = np.array([130, 252, 254],np.uint8)

    # means_h2
    COLOR_MIN = np.array([55, 143, 168],np.uint8)
    COLOR_MAX = np.array([111, 255, 181],np.uint8)    

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Apply threshold for image segmentation
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    return frame_threshed

# Function to perform morphological operations on an image
def morf_operations(image, kernel):
    HSV_th = HSV_threshold(image)
    cv2.imshow("HSV", HSV_th)
    cv2.waitKey(0)

    # Apply dilation operation
    opening_img = morf.dilate(HSV_th, kernel)
    cv2.imshow("OP", opening_img)
    cv2.waitKey(0)

    # Create a new kernel
    kernel = morf.create_kernel(4, 'ellipse')

    # Apply dilation operation with the new kernel
    opening_img = morf.dilate(opening_img, kernel)
    cv2.imshow("OP1", opening_img)
    cv2.waitKey(0)

    # Commented - different morphological operations
    # kernel = morf.create_kernel(3, 'square')
    # dilate_img = morf.opening(opening_img, kernel)
    # cv2.imshow("Dilate", dilate_img)
    # cv2.waitKey(0)

    # means4
    # opening_img = morf.opening(HSV_th, kernel)
    # cv2.imshow("OP", opening_img)
    # cv2.waitKey(0)

    # dilate_img = morf.dilate(opening_img, kernel)
    # cv2.imshow("Dilate", dilate_img)
    # cv2.waitKey(0)
    
    return opening_img

# Function to apply Canny edge detection on an image
def Canny(image):
    frame = cv2.GaussianBlur(image, (5,5), 0) # Reduces noise

    cv2.imshow("gaussian_b", frame)
    cv2.waitKey(0)

    edges = cv2.Canny(frame, 50, 200, None, 3)

    cv2.imshow("Canny", edges)
    cv2.waitKey(0)
    return edges

# Function to create a mask for the image
def Create_mask(image):
    w = image.shape[0]
    h = image.shape[1]
    top_rec_x, top_rec_y = 0, 230
    top_rec_x2, top_rec_y2 = h,230

    print("Width: ", w)
    print("Height: ", h)
    tmask_points = np.array([(top_rec_x, top_rec_y), (top_rec_x2, top_rec_y2), (h, w), (0, w)])

    # Create a mask
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, [tmask_points], (255, 255, 255))
    masked_img = cv2.bitwise_and(image, mask)
    # plt.imshow(masked_img)
    # plt.show()
    cv2.imshow("Mask", masked_img)
    cv2.waitKey(0)
    return masked_img

# Function to calculate the coordinates of a line based on its slope and intercept
def cordinates(image, lane):
    slope, intersec = lane
    y1 = image.shape[0]
    y2 = int(y1*(3/4))
    x1 = int((y1 - intersec)/slope)
    x2 = int((y2 - intersec)/slope)
    return np.array([x1,y1,x2,y2])

# Function to find the average line from the detected lines
def unique_line(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2, y2 = line.reshape(4)
            print(line)
            param = np.polyfit((x1,x2), (y1, y2), 1) # Polyfit returns an array of [(slope), intersection]
            slope = param[0]
            intersection = param[1] 
            if slope < 0: # Slope < 0 represents a left line
                left_lines.append((slope, intersection))
            if slope > 0: # Slope > 0 represents a right line
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
    
# Function to draw lines on the image
def draw_lines(image, Lines):
    road = np.zeros_like(image)
    if Lines is not None:
        for line in Lines:
            x1,y1,x2,y2 = line.reshape(4)
            if(abs(y1-y2) > 10): # Avoid drawing horizontal lines
                cv2.line(road, (x1,y1), (x2, y2), color=(0, 255, 255), thickness=3)
                cv2.line(image, (x1,y1), (x2, y2), color=(0, 255, 255), thickness=3)
    return road

def main():
    img_background = cv2.imread('ic_images/um_000008.png')
    img_main = cv2.imread('ic_images/clust+hsv+morf/mean_shift++/mask/segmentation42.png')
    # img_main = img_background
  
    if img_main is None or img_background is None:
        print("Image not found")
        exit()

    cv2.imshow("Resized_background", img_background)
    cv2.waitKey(0)
    cv2.imshow("Resized_main", img_main)
    cv2.waitKey(0)

    # Limit the vision of the horizon
    mask_img = Create_mask(img_main)

    # Treatment of the image by removing undesired pixels
    img_morf = morf_operations(mask_img, morf.create_kernel(4, 'ellipse'))

    # Overlay the road on the background image
    road_seg = cv2.cvtColor(img_morf, cv2.COLOR_GRAY2BGR)
    road_seg[img_morf == 255] = [255, 0, 0]
    cv2.imshow("Road", road_seg)
    cv2.waitKey(0)

    result = cv2.addWeighted(img_background, 0.8, road_seg, 0.9, 1)
        
    # Showing results
    cv2.imshow("Frames", result)
    cv2.waitKey(0) 

if __name__ == "__main__":
    main()
