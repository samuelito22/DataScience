# Image Processing Script
`main.py` is a script that enhances images using various techniques, such as Laplacian edge detection, median filtering, maximum filtering, perspective transformations, power-law transformations, brightness adjustments, and histogram equalization. The script processes images concurrently using ThreadPoolExecutor.

## Functions
1. laplacianEdge(img): Applies Laplacian edge detection to the input image.
2. filterImage(img, kernel_size): Applies median and maximum filters to the input image.
3. wrapPerspective(img, distance): Applies a perspective transformation to the input image with four specified points.
4. power_law_transform(img, gamma): Applies a power-law transformation to the input image with a specified gamma value.
5. brightness_adjustment(img, factor): Adjusts the input image's brightness using a specified factor.
6. histEqualization(img, clipLimit): Applies histogram equalization to the input image with a specified clip limit.
7. improvedImage(img, mask): Enhances the input image using a combination of the above techniques.
8. process_image(file): Reads an image file, enhances it, and saves it to the "Results" folder.
Usage

Run the script by executing the following command in your terminal:

```bash 
python main.py <path_to_test_images>
```
Replace <path_to_test_images> with the path to the folder containing test images.

The script processes the images and saves enhanced versions in a folder called "Results".

## Dependencies
The script requires the following Python libraries:

- OpenCV (cv2)
- NumPy
- argparse
- os
- concurrent.futures
Ensure these libraries are installed before running the script.