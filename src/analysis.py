import cv2
from ultralytics import YOLO
import os
import csv
import time

# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("models/yolo11n.pt")

# ==========================================
# 2. VIDEO PATHS
# ==========================================

input_video = "input/futsal_match.mp4"
output_video = "output/futsal_analysis.mp4"
csv_file = "results/player_statistics.csv"

# ==========================================
# 3. CHECK INPUT
# ==========================================

if not os.path.exists(input_video):
    print("ERROR: Futsal video not found!")
    print(input_video)
    exit()

# Create results folder if necessary
os.makedirs("results", exist_ok=True)

# ==========================================
# 4. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("===================================")
print("FUTSAL MATCH ANALYSIS")
print("===================================")
print(f"Video Width: {width}")
print(f"Video Height: {height}")
print(f"Video FPS: {fps}")
print(f"Total Frames: {total_frames}")
print("Running analysis on CPU...")
print("===================================")

# ==========================================
# 5. CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# ==========================================
# 6. CSV FILE
# ==========================================

csv_output = open(
    csv_file,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_output)

csv_writer.writerow([
    "Frame",
    "Time_Seconds",
    "Player_Count",
    "Ball_Detected"
])

# ==========================================
# 7. PROCESS VIDEO
# ==========================================

frame_count = 0
start_time = time.time()

max_players_seen = 0
ball_frames = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # ======================================
    # YOLO TRACKING
    # ======================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device="cpu",
        conf=0.20,
        imgsz=960,
        verbose=False
    )

    result = results[0]

    player_count = 0
    ball_detected = False

    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            # ----------------------------------
            # PLAYER
            # ----------------------------------

            if class_name == "person":

                player_count += 1

                if box.id is not None:
                    track_id = int(box.id[0])
                else:
                    track_id = -1

                # Draw player box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Player label
                label = f"Player {track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # ----------------------------------
            # FOOTBALL
            # ----------------------------------

            elif class_name == "sports ball":

                ball_detected = True

                # Draw ball box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                ball_label = f"Football {confidence:.2f}"

                cv2.putText(
                    frame,
                    ball_label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

    # ======================================
    # UPDATE STATISTICS
    # ======================================

    if player_count > max_players_seen:
        max_players_seen = player_count

    if ball_detected:
        ball_frames += 1

    # ======================================
    # TIME
    # ======================================

    time_seconds = frame_count / fps

    # ======================================
    # SAVE CSV DATA
    # ======================================

    csv_writer.writerow([
        frame_count,
        round(time_seconds, 2),
        player_count,
        int(ball_detected)
    ])

    # ======================================
    # DISPLAY INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"Players: {player_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Ball: {'Detected' if ball_detected else 'Not Detected'}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ======================================
    # FPS
    # ======================================

    elapsed = time.time() - start_time

    current_fps = frame_count / elapsed

    cv2.putText(
        frame,
        f"Processing FPS: {current_fps:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "FUTSAL MATCH ANALYSIS",
        frame
    )

    # ======================================
    # SAVE VIDEO
    # ======================================

    out.write(frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Analysis stopped by user.")
        break


# ==========================================
# 8. CLOSE FILES
# ==========================================

cap.release()
out.release()
csv_output.close()
cv2.destroyAllWindows()

# ==========================================
# 9. FINAL RESULTS
# ==========================================

print("===================================")
print("ANALYSIS COMPLETE!")
print("===================================")
print(f"Processed Frames: {frame_count}")
print(f"Maximum Players Detected: {max_players_seen}")
print(f"Frames With Ball Detection: {ball_frames}")
print()
print("Output Video:")
print(output_video)
print()
print("CSV Statistics:")
print(csv_file)
print("===================================")


