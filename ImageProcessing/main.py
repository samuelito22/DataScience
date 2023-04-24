import cv2
import numpy as np
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

# Applies the Laplacian edge detection filter to the input image
def laplacianEdge(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian filter to the grayscale image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    # Convert the Laplacian output to the absolute value and convert it to a 3-channel image
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    laplacian_3channels = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)

    # Add the Laplacian output to the original image with a negative weight to create the filtered image
    filtered_img =cv2.addWeighted(img, 1, laplacian_3channels, -0.47, 0)

    return filtered_img

# Applies a median filter, a maximum filter, and a warp perspective transformation to the input image
def filterImage(img, kernel_size):
    # Apply a median filter to the input image
    median_filtered = cv2.medianBlur(img, 3)

    # Apply a maximum filter (dilation) to the median-filtered image
    max_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    max_filtered = cv2.dilate(median_filtered, max_kernel)
    
    return max_filtered

# Applies a perspective transformation to the input image using four specified points
def wrapPerspective(img, distance):
    # Define the four points of the original image and the four points of the output image
    topLeft = [57,13] 
    topRight = [185,7] 
    bottomLeft = [80,239]
    bottomRight = [201,233] 
    point_matrix = np.float32([topLeft,topRight,bottomLeft,bottomRight])
    height, width = img.shape[:2]
    converted_topLeft = [(width/2) - (distance/2),0]
    converted_topRight = [(width/2) + (distance/2),0]
    converted_bottomLeft = [(width/2) - (distance/2),height]
    converted_bottomRight = [(width/2) + (distance/2),height]    
    converted_points = np.float32([converted_topLeft,converted_topRight, converted_bottomLeft,converted_bottomRight])

    # Apply the perspective transformation to the input image using the specified points
    perspective_transform = cv2.getPerspectiveTransform(point_matrix,converted_points)
    img_Output = cv2.warpPerspective(img,perspective_transform,(width,height))
    return img_Output

# Applies a power-law transformation to the input image using a specified gamma value
def power_law_transform(img, gamma):
    return np.uint8(np.power(img/255.0, gamma)*255)

# Adjusts the brightness of the input image using a specified factor
def brightness_adjustment(img, factor):
    return np.clip((img * factor).astype(int), 0, 255).astype(np.uint8)

# Applies histogram equalization to the input image using a specified clip limit
def histEqualization(img, clipLimit):
    clahe = cv2.createCLAHE(clipLimit = clipLimit )

    # Apply histogram equalization to each color channel of the input image
    colorimage_b = clahe.apply(img[:,:,0])
    colorimage_g = clahe.apply(img[:,:,1])
    colorimage_r = clahe.apply(img[:,:,2])

    # Merge the equalized color channels back into a single
    return cv2.merge((colorimage_b,colorimage_g, colorimage_r))

def improvedImage(img, mask):
    improvedImage = img.copy()
    # Fill in missing region of image
    improvedImage = cv2.inpaint(improvedImage,mask,5,cv2.INPAINT_NS)

    improvedImage = wrapPerspective(improvedImage, 141)

    improvedImage = filterImage(improvedImage, 3)
    
    improvedImage = laplacianEdge(improvedImage)

    improvedImage = histEqualization(improvedImage, clipLimit=1.53)

    improvedImage = power_law_transform(improvedImage, 0.5)

    improvedImage = brightness_adjustment(improvedImage, 0.62)

    return improvedImage

def process_image(file):
    img = cv2.imread(os.path.join(args.data, file))
    if img is not None:
        enhanced_img = improvedImage(img, mask)
        cv2.imwrite(os.path.join("Results", file), enhanced_img)

parser = argparse.ArgumentParser(description='Processing images')
parser.add_argument("data", type=str, help="specify path to test images", default='test_images')
args = parser.parse_args()

mask = cv2.imread("mask.png", 0)

if not os.path.exists("Results"):
    os.makedirs("Results")

with ThreadPoolExecutor() as executor:
    list(executor.map(process_image, os.listdir(args.data)))