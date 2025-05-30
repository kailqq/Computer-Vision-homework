import cv2
import numpy as np
import argparse
import sys
import os
import glob

def calibrate_camera_from_images(board_width, board_height, images_dir, save_file):
    """
    Calibrate camera from images in a folder.
    """
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    # Get all image files in the specified directory
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))
    
    if not image_files:
        print(f"Error: No image files found in directory '{images_dir}'.")
        return False
    
    print(f"\nFound {len(image_files)} image files. Processing...")
    # Create window for display
    cv2.namedWindow('Processing Image', cv2.WINDOW_NORMAL)
    # Image size variable for calibration
    image_size = None
    # Process each image file
    successful_images = 0
    for idx, image_path in enumerate(image_files):
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Cannot read image '{image_path}', skipping.")
            continue
        # Save size from the first valid image
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)
        # Prepare display image
        display_img = img.copy()
        # If found corners
        if ret:
            # Refine corner detection
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            # Save results
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images += 1
            # Draw corners on image
            cv2.drawChessboardCorners(display_img, (board_width, board_height), corners2, ret)
            status = "Success: Chessboard detected"
            color = (0, 255, 0)  # Green
        else:
            status = "Fail: Chessboard not detected"
            color = (0, 0, 255)  # Red
        # Show progress and status
        cv2.putText(display_img, f"Processing: {idx+1}/{len(image_files)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_img, status, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_img, f"Success: {successful_images}/{len(image_files)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        # Show current image name
        file_name = os.path.basename(image_path)
        if len(file_name) > 25:  # Truncate if too long
            file_name = file_name[:22] + "..."
        cv2.putText(display_img, file_name, (10, display_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Show image
        cv2.imshow('Processing Image', display_img)
        cv2.imwrite(f"{images_dir}/output/output_with_corners_{idx+1}.jpg", display_img)
        key = cv2.waitKey(5000) & 0xFF  # Show briefly then continue
        if key == 27:  # ESC
            print("User interrupted processing.")
            cv2.destroyAllWindows()
            return False
    cv2.destroyAllWindows()
    if successful_images == 0:
        print("Error: No valid chessboard images found.")
        return False
    print(f"\nSuccessfully processed {successful_images} images, starting camera calibration...")
    # Camera calibration
    if image_size is None:  # Safety check
        print("Error: Cannot determine image size.")
        return False
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    
    if not ret:
        print("Camera calibration failed.")
        return False
        
    print("Camera calibration successful!")
    print("Camera matrix (mtx):\n", mtx)
    print("Distortion coefficients (dist):\n", dist)
    # Compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\nMean reprojection error: {mean_error / len(objpoints)}")
    # Save calibration result
    np.savez(save_file, mtx=mtx, dist=dist)
    print(f"\nCalibration data saved to '{save_file}'")
    return True


def analyze_image(board_width, board_height, load_file, image_path):
    """
    Load calibration data, analyze image, perform pose estimation and bird's eye view transformation.
    """
    # Load calibration data
    try:
        with np.load(load_file) as data:
            mtx = data['mtx']
            dist = data['dist']
    except FileNotFoundError:
        print(f"Error: Calibration file '{load_file}' not found. Please run calibration first.")
        return

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image '{image_path}'.")
        return
        
    print(f"\nAnalyzing image: '{image_path}'")
    
    # Prepare object points
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, (board_width, board_height), None)

    if not ret:
        print("Chessboard not found in image.")
        return

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # --- 1. Pose Estimation ---
    # Use solvePnP to get rotation and translation vectors
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    print("\n--- Pose Estimation Result ---")
    print("Rotation vector (rvec):\n", rvec)
    print("Translation vector (tvec) (unit same as chessboard square size):\n", tvec)
    print("Rotation matrix (rmat):\n", rmat)

    # Draw axes on image for visualization
    axis_length = 3.0
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
    print('axis_points',axis_points)
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, mtx, dist)

    # Draw axes
    img_with_axis = img.copy()
    origin = tuple(img_points[0].ravel().astype(int))
    cv2.line(img_with_axis, origin, tuple(img_points[1].ravel().astype(int)), (255, 0, 0), 3)  # X: Blue
    cv2.line(img_with_axis, origin, tuple(img_points[2].ravel().astype(int)), (0, 255, 0), 3)  # Y: Green
    cv2.line(img_with_axis, origin, tuple(img_points[3].ravel().astype(int)), (0, 0, 255), 3)  # Z: Red


    # --- 2. Bird's Eye View Transformation ---
    # Four corresponding corner points in the image
    img_pts = np.float32([
        corners2[0].ravel(),
        corners2[board_width - 1].ravel(),
        corners2[(board_height - 1) * board_width].ravel(),
        corners2[board_width * board_height - 1].ravel()
    ])

    # Define output size
    h, w = img.shape[:2]
    
    # --- Add interactive bird's eye view generation ---
    print("\n--- Bird's Eye View Transformation ---")
    print("\nInteractive Bird's Eye View:")
    print("- Press 'u' to increase virtual height")
    print("- Press 'd' to decrease virtual height")
    print("- Press 'r' to reset view")
    print("- Press 's' to save the image")
    print("- Press 'ESC' to exit")
    
    # Create and resize windows
    cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Estimation', 600, 600)
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 600, 600)
    cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bird's Eye View", 600, 600)
    # Display camera pose and original image
    cv2.imshow('Pose Estimation', cv2.resize(img_with_axis, (400, 400)))
    cv2.imshow('Original Image', img)


    # Initial "height" parameter
    Z = 25.0
    Z_change = 0.5  # Step size for each adjustment
    
    # Define target area (centered) - calculate margins to center the chessboard
    margin_x = int(w * 0.2)  # Horizontal margin 20%
    margin_y = int(h * 0.2)  # Vertical margin 20%

    
    # Interactive loop
    while True:
        # Generate target points - resize but keep centered
        # The larger the Z value, the smaller the chessboard appears (further "away")
        scale_factor = 25.0 / Z  # Base value is 25
        
        # Calculate scaled target area (keep centered)
        center_x, center_y = w/2, h/2
        half_width = (w/2 - margin_x) * scale_factor
        half_height = (h/2 - margin_y) * scale_factor
        
        scaled_dst_pts = np.float32([
            [center_x - half_width, center_y - half_height],  # Top-left
            [center_x + half_width, center_y - half_height],  # Top-right
            [center_x - half_width, center_y + half_height],  # Bottom-left
            [center_x + half_width, center_y + half_height]   # Bottom-right
        ])
        
        # Calculate homography matrix from image points to target points
        H_direct = cv2.getPerspectiveTransform(img_pts, scaled_dst_pts)
        # Compute transformed image
        bird_eye_view = cv2.warpPerspective(
            img, 
            H_direct,  # Use directly calculated transformation matrix
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Display current Z value
        status_img = bird_eye_view.copy()
        cv2.putText(
            status_img, 
            f"Z = {Z:.1f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), 
            2
        )
        
        # Display image
        cv2.imshow("Bird's Eye View", status_img)
        # Wait for keyboard input
        key = cv2.waitKey(100) & 0xFF
        # Adjust Z value based on keyboard input
        if key == ord('u'):  # Increase "height"
            Z += Z_change
        elif key == ord('d'):  # Decrease "height"
            Z = max(0.1, Z - Z_change)  # Prevent Z from becoming negative or zero
        elif key == ord('r'):  # Reset
            Z = 25.0
        elif key == ord('s'): #save image
            image_name = os.path.basename(args.image).split('.')[0]
            cv2.imwrite(f'./birdseye/BirdsViewOf_{image_name}.jpg', status_img)

        elif key == 27:  # ESC key
            break
            
    cv2.destroyAllWindows()
    print("\nAnalysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera calibration and pose estimation in Python.")
    parser.add_argument('-W', '--width', type=int, required=True, help="Number of inner corners in chessboard width.")
    parser.add_argument('-H', '--height', type=int, required=True, help="Number of inner corners in chessboard height.")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--calibrate', action='store_true', help="Run camera calibration mode.")
    group.add_argument('--analyze', action='store_true', help="Run image analysis mode.")
    
    # Calibration parameters
    parser.add_argument('--images_dir', type=str, help="Path to folder containing calibration images.")
    parser.add_argument('--savefile', type=str, default='camera_data.npz', help="Path to save calibration data.")

    # Analysis parameters
    parser.add_argument('--loadfile', type=str, default='camera_data.npz', help="Path to load calibration data.")
    parser.add_argument('--image', type=str, help="Path to chessboard image for analysis.")

    args = parser.parse_args()

    if args.calibrate:
        if not args.images_dir:
            print("Error: --images_dir parameter is required in calibration mode.")
            sys.exit(1)
        calibrate_camera_from_images(args.width, args.height, args.images_dir, args.savefile)
    elif args.analyze:
        if not args.image:
            print("Error: --image parameter is required in analysis mode.")
            sys.exit(1)
        analyze_image(args.width, args.height, args.loadfile, args.image)
