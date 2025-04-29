import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

def read_image_and_fit_ellipse(image_path):
    """
    Read an image, detect edges, and fit ellipses
    Parameters:
        image_path: Path to the image file
    """
    print(f"Processing image: {image_path}")
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a copy for drawing results
    result_img = img.copy()
    # Iterate through all contours
    count_fitted = 0
    for contour in contours:
        # Only process large enough contours (filter noise)
        if len(contour) >= 5 and cv2.contourArea(contour) > 100:
            # Fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(result_img, ellipse, (0, 0, 255), 2)
                # Get the rotated rectangle (bounding box) of the ellipse
                box = cv2.boxPoints(ellipse)
                box = np.intp(box)
                cv2.drawContours(result_img, [box], 0, (255, 0, 0), 2)
                count_fitted += 1
            except:
                print(f"Error fitting ellipse to contour with {len(contour)} points")
    print(f"Fitted ellipses to {count_fitted} contours")
    # Create output directory structure
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.getcwd(), "output")
    image_output_dir = os.path.join(output_dir, base_filename)
    # Ensure both the output dir and the image-specific subdir exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    # Save the result images
    edges_path = os.path.join(image_output_dir, f"{base_filename}_edges.jpg")
    result_path = os.path.join(image_output_dir, f"{base_filename}_ellipse_result.jpg")
    # Save images and verify they were saved correctly
    cv2.imwrite(edges_path, edges)
    cv2.imwrite(result_path, result_img)

def main():
    if len(sys.argv) != 2:
        print("No input directory detected")
        print("Use: 'python hw2.py hw2_data' to process input data")
        return
    
    input_dir = sys.argv[1]
    print(f"Looking for images in: {input_dir}")
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    # Get all image files from the input directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                 glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        print(f"No image files found in directory: {input_dir}")
        return
    print(f"Found {len(image_files)} image files to process")
    for image_file in image_files:
        read_image_and_fit_ellipse(image_file)
if __name__ == "__main__":
    main()