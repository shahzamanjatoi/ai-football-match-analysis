import cv2
from ultralytics import YOLO
import csv
import os

model = YOLO("models/yolo11n.pt")

input_video = "input/futsal_match.mp4"
output_video = "output/futsal_tracking_ids.mp4"
csv_file = "results/tracking_ids.csv"

os.makedirs("output", exist_ok=True)
os.makedirs("results", exist_ok=True)

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

csv_output = open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_output)

writer.writerow([
    "Frame",
    "Time_Seconds",
    "Tracking_ID",
    "X1",
    "Y1",
    "X2",
    "Y2"
])

# Store all IDs found
all_ids = set()

frame_number = 0

print("===================================")
print("TRACKING ID CHECK")
print("===================================")
print(f"Total Frames: {total_frames}")
print("Processing...")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device="cpu",
        conf=0.15,
        imgsz=960,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Only players
            if class_name != "person":
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            all_ids.add(track_id)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            time_seconds = frame_number / fps

            writer.writerow([
                frame_number,
                round(time_seconds, 2),
                track_id,
                x1,
                y1,
                x2,
                y2
            ])

            # Draw player box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display Tracking ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Frame information
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique IDs so far: {len(all_ids)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    out.write(frame)

    cv2.imshow(
        "Tracking ID Check",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
csv_output.close()
cv2.destroyAllWindows()

print()
print("===================================")
print("TRACKING COMPLETE")
print("===================================")
print(f"Processed Frames: {frame_number}")
print(f"Total Unique Tracking IDs: {len(all_ids)}")

print()
print("ALL TRACKING IDs:")
print("-----------------------------------")

for track_id in sorted(all_ids):
    print(f"ID {track_id}")

print()
print("Output Video:")
print(output_video)

print()
print("CSV:")
print(csv_file)

print("===================================")
