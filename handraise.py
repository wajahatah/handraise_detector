"""from ultralytics import YOLO
import os
import cv2
import csv

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")

    # Open the video file
    video_path = "Cam_19_14.mp4"
    cap = cv2.VideoCapture(video_path)

    # Prepare output paths
    output_folder = "frames_output/video1"
    os.makedirs(output_folder, exist_ok=True)
    csv_file_path = os.path.join(output_folder, "video1nn_output.csv")

    # Create the CSV file with headers
    headers = ["Frame Name"] + [f"person{i}{chr(k)}" for i in range(0, 5) for k in range(ord('a'), ord('j') + 1)]
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

    # Check if the video capture opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0  # To track frame number

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Generate frame folder and frame name
        frame_name = f"frame_{frame_count:04d}"
        frame_folder = os.path.join(output_folder, frame_name)
        os.makedirs(frame_folder, exist_ok=True)

        # Run inference on the current frame
        results = model(frame)

        # Initialize a row for the CSV
        csv_row = [frame_name]

        # Iterate over each detected object and extract keypoints
        for result in results:
            keypoints = result.keypoints  # Access the keypoints object

            if keypoints is not None:
                # Get the data attribute, which contains x, y, and confidence values
                keypoints_data = keypoints.data
                for person_keypoints in keypoints_data:
                    # For each person, save keypoints as x, y, confidence
                    person_data = []
                    print("keypoints", person_data, "2",keypoints_data)
                    for keypoint in person_keypoints:
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        person_data.append(f"({x:.2f}, {y:.2f}, {confidence:.2f})")
                    # Append person's keypoints to the CSV row
                    csv_row.extend(person_data)

        # Fill empty cells in the row if fewer persons are detected
        while len(csv_row) < len(headers):
            csv_row.append("")

        # Write the row to the CSV
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_row)

        # Save the frame with keypoints drawn
        frame_output_path = os.path.join(frame_folder, f"{frame_name}.png")
        cv2.imwrite(frame_output_path, frame)
        # cv2.imshow(frame)

        frame_count += 1  # Increment frame counter

    # Release the video capture and close display window
    cap.release()
    print("Processing complete. Frames and CSV saved.")
"""

from ultralytics import YOLO
import os
import cv2
import csv

# Avoid potential library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the bounding ranges for different persons
dict1 = {'xmin': 60, 'ymin': 90, 'xmax': 367, 'ymax': 400}
dict2 = {'xmin': 370, 'ymin': 90, 'xmax': 730, 'ymax': 400}
dict3 = {'xmin': 740, 'ymin': 90, 'xmax': 1075, 'ymax': 400}
dict4 = {'xmin': 1080, 'ymin': 90, 'xmax': 1280, 'ymax': 400}

ranges = [dict1, dict2, dict3, dict4]

if __name__ == "__main__":
    # Load your trained YOLOv8 model
    model = YOLO("runs/pose/trail4/weights/best_y11.pt")

    # Open the video file
    video_path = "C:/Users/LAMBDA THETA/Downloads/handraise/cb4-65.mp4" #"Cam_19_14.mp4"
    # video_path = "C:/Users/LAMBDA THETA/Downloads/handraise/hr2.mp4" #"Cam_19_14.mp4"
    cap = cv2.VideoCapture(video_path)

    # Prepare output paths
    output_folder = "frames_output/video1"
    os.makedirs(output_folder, exist_ok=True)
    csv_file_path = os.path.join(output_folder, "n.csv")

    # Create the CSV file with headers
    headers = ["Frame Name"] + [f"person{i}{chr(k)}" for i in range(1, 5) for k in range(ord('a'), ord('j') + 1)]
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

    # Check if the video capture opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0  # To track frame number

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available

        # Generate frame folder and frame name
        frame_name = f"frame_{frame_count:04d}"
        # frame_folder = os.path.join(output_folder, frame_name)
        # os.makedirs(frame_folder, exist_ok=True)

        # Run inference on the current frame
        frame = cv2.resize(frame,(1280,720))
        results = model(frame)

        # Initialize a row for the CSV
        csv_row = [frame_name]
        person_columns = [[] for _ in range(4)]  # List to store keypoints for each person column

        # Iterate over each detected object and extract keypoints
        for result in results:
            keypoints = result.keypoints  # Access the keypoints object

            if keypoints is not None:
                # Get the data attribute, which contains x, y, and confidence values
                keypoints_data = keypoints.data
                for person_keypoints in keypoints_data:
                    # Check where the keypoints belong based on their ranges
                    for keypoint in person_keypoints:
                        x, y, confidence = keypoint[0].item(), keypoint[1].item(), keypoint[2].item()
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                        cv2.putText(
                                frame,
                                f"({int(x)}, {int(y)})",
                                (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 0, 0),
                                1
                            )

                        # Determine the person column based on the ranges
                        for idx, r in enumerate(ranges):
                            if r['xmin'] <= x <= r['xmax'] and r['ymin'] <= y <= r['ymax']:
                                person_columns[idx].append(f"({x:.2f}, {y:.2f}, {confidence:.2f})")
                                break

        # Fill the CSV row with detected keypoints in respective columns
        for person_data in person_columns:
            csv_row.extend(person_data)
            # Fill missing keypoints with "N/A" to maintain the format
            csv_row.extend(["0"] * (9 - len(person_data)))

        # Write the row to the CSV
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_row)

        # Save the frame with keypoints drawn
        frame_output_path = os.path.join(output_folder, f"{frame_name}.png")
        cv2.imwrite(frame_output_path, frame)

        # Show saved frame and CSV row
        # print(f"Frame saved: {frame_output_path}")
        # print(f"CSV row saved: {csv_row}")

        frame_count += 1  # Increment frame counter

    # Release the video capture and close display window
    cap.release()
    print(f"Processing complete. Frames and CSV saved in: {output_folder}")
