import cv2
import numpy as np

# Define a function to process the video and extract the trajectory
def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        return None, "Could not open video"
    
    # Read the first frame of the video
    ret, first_frame = cap.read()
    if not ret:
        return None, "Could not read the first frame"
    
    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Create a mask image for drawing (in color)
    mask = np.zeros_like(first_frame)
    
    while cap.isOpened():
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the new frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to RGB (color) image
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Update previous frame
        prev_gray = gray
    
    # Release the video capture object
    cap.release()
    
    # Return the trajectory image
    return rgb, "Processing completed"

# Path to the uploaded video file
video_path = './example_video.mp4'

# Process the video and get the trajectory image
trajectory_image, message = process_video(video_path)

# Save the trajectory image
if trajectory_image is not None:
    cv2.imwrite('./trajectory_image.png', trajectory_image)

# Return the result
trajectory_image_path = './trajectory_image.png' if trajectory_image is not None else None
(trajectory_image_path, message)
