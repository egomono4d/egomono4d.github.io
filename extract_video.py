import cv2
import os

def extract_frames_opencv(video_path, output_dir):
    # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames or an error occurred

        # Construct output filename (e.g., frame_00001.jpg)
        out_filename = os.path.join(output_dir, f"frame_{frame_index:05d}.jpg")
        # cv2.imwrite(out_filename, frame)
        print(out_filename)
        frame_index += 1

    cap.release()
    print(f"Extraction complete. Total frames: {frame_index}")

if __name__ == "__main__":
    extract_frames_opencv("recordings/4ed63fb1.mp4", "frames_opencv")