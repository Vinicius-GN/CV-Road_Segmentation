import cv2
import numpy as np

def erode(img, kernel):
    img = cv2.erode(img, kernel, iterations=1)
    return img

def dilate(img, kernel):
    img = cv2.dilate(img, kernel, iterations=1)
    return img

def opening(img, kernel):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def closing(img, kernel):
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def create_kernel(size, shape):
    if shape == 'square':
        return np.ones((size, size), np.uint8)
    if shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
        return kernel
    if shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return kernel
    else:
        print("Invalid shape")
        return None
    
def main():
    img = cv2.imread("Images/processed.png")

    cv2.imshow("Original_img", img)
    cv2.waitKey(0)

    structuring_element = create_kernel(3, 'square')
    img = opening(img, structuring_element)

    cv2.imshow("Opening", img)
    cv2.waitKey(0)


    structuring_element = create_kernel(4, 'square')
    img = opening(img, structuring_element)

    cv2.imshow("Opening 2", img)
    cv2.waitKey(0)

    img = dilate(img, structuring_element)

    cv2.imshow("Dilate", img)
    cv2.waitKey(0)

    
    

#if __name__ == "__main__":
#    main()