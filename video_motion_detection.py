import cv2
import mediapipe as mp
import os

# Input and output paths
video_path = r"C:\Users\abdul raheman\OneDrive\python\Naresh_it\python\YOLO\yolo video detection\man dancing.mp4"
output_folder = r"C:\Users\abdul raheman\OneDrive\python\Naresh_it\python\YOLO\yolo video detection\output video"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "motion_detected_output.avi")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Read video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"⚠️ Video not found: {video_path}")

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)  # Use float FPS for accuracy
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible than MJPG
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("🎬 Processing video, please wait...")

# Process each frame
frame_index = 0
while frame_index < frame_count:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    out.write(frame)  # Save frame to output video
    frame_index += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Automatically open the saved video
os.startfile(output_video_path)
print(f"✅ Motion detection video saved at:\n{output_video_path}")
