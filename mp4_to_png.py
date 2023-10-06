import cv2
import os



def extract_frames(input_file, output_dir):
    # Create a video capture object
    cap = cv2.VideoCapture(input_file)
    frame_count = 0

    # Read frames and save them
    png_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frames every 1 second
        if frame_count % cap.get(cv2.CAP_PROP_FPS) == 0:
            frame_filename = os.path.join(output_dir, f"frame_{png_cnt}.png")
            png_cnt += 1
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the capture object
    cap.release()

if __name__ == "__main__":
    input_file = "movie/zunda_tesla_720p.mp4"  # Specify the input file path
    output_dir = "frames_720p"    # Directory for saving frame images

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_frames(input_file, output_dir)
