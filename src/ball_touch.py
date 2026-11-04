import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

INPUT_VIDEO = "input/futsal_match.mp4"
OUTPUT_VIDEO = "output/futsal_uzair_touch.mp4"
OUTPUT_CSV = "results/uzair_touch_results.csv"

MODEL_PATH = "models/yolo11n.pt"

# Uzair's ByteTrack ID
UZAIR_ID = 13

# Detection settings
CONFIDENCE = 0.15
IMAGE_SIZE = 960

# Maximum distance between ball and player for possible touch
TOUCH_DISTANCE = 50

# Number of consecutive frames required
MIN_CONSECUTIVE_FRAMES = 2

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("----------------------------------------")
print("VIDEO INFORMATION")
print("----------------------------------------")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps:.2f}")
print(f"Total Frames: {total_frames}")
print("----------------------------------------")

# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# VARIABLES
# ============================================================

results_data = []

frame_number = 0

# Consecutive frames where Uzair is close to ball
uzair_close_frames = 0

# Whether we are currently inside one touch event
touch_active = False

# Total separate touch events
uzair_touch_count = 0

# Used to display "UZAIR TOUCH!" temporarily
touch_message_frames = 0

# Number of frames to display touch message
TOUCH_MESSAGE_DURATION = int(fps * 0.8)

# Store current touch start information
current_touch_start_frame = None
current_touch_start_time = None

# ============================================================
# PROCESS VIDEO
# ============================================================

print("\nProcessing video...")
print("Press Q to stop.\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        device="cpu",
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    player_boxes = []
    ball_box = None

    # --------------------------------------------------------
    # READ DETECTIONS
    # --------------------------------------------------------

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for i in range(len(boxes)):

            cls_id = int(boxes.cls[i])

            confidence = float(boxes.conf[i])

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i].cpu().numpy()
            )

            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if cls_id == 0:

                track_id = None

                if boxes.id is not None:
                    track_id = int(boxes.id[i])

                if track_id is not None:

                    player_boxes.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2),
                        "confidence": confidence
                    })

            # ------------------------------------------------
            # SPORTS BALL
            # ------------------------------------------------

            elif cls_id == 32:

                # YOLO COCO class 32 = sports ball

                ball_box = (x1, y1, x2, y2)

    # ========================================================
    # DRAW ALL PLAYERS
    # ========================================================

    uzair_found = False
    uzair_box = None

    for player in player_boxes:

        player_id = player["id"]

        x1, y1, x2, y2 = player["box"]

        # ----------------------------------------------------
        # UZAIR
        # ----------------------------------------------------

        if player_id == UZAIR_ID:

            uzair_found = True
            uzair_box = (x1, y1, x2, y2)

            # Draw Uzair box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            # Name
            cv2.putText(
                frame,
                "UZair (ID 13)",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            # Other players
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"ID {player_id}",
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    # ========================================================
    # BALL
    # ========================================================

    ball_distance = None
    likely_touch = False

    if ball_box is not None:

        bx1, by1, bx2, by2 = ball_box

        # Ball center
        ball_center_x = int((bx1 + bx2) / 2)
        ball_center_y = int((by1 + by2) / 2)

        # Draw ball box
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 0, 255),
            2
        )

        cv2.circle(
            frame,
            (ball_center_x, ball_center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "BALL",
            (bx1, max(by1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # ====================================================
        # CHECK DISTANCE FROM UZAIR TO BALL
        # ====================================================

        if uzair_found and uzair_box is not None:

            ux1, uy1, ux2, uy2 = uzair_box

            # Find nearest point on Uzair's bounding box
            nearest_x = max(ux1, min(ball_center_x, ux2))
            nearest_y = max(uy1, min(ball_center_y, uy2))

            distance = np.sqrt(
                (ball_center_x - nearest_x) ** 2 +
                (ball_center_y - nearest_y) ** 2
            )

            ball_distance = distance

            # Draw line between Uzair and ball
            cv2.line(
                frame,
                (ball_center_x, ball_center_y),
                (nearest_x, nearest_y),
                (0, 255, 255),
                2
            )

            # Display distance
            cv2.putText(
                frame,
                f"Distance: {distance:.1f}px",
                (ux1, min(uy2 + 25, height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # =================================================
            # POSSIBLE TOUCH
            # =================================================

            if distance <= TOUCH_DISTANCE:

                uzair_close_frames += 1

            else:

                uzair_close_frames = 0

                # Reset touch state when ball moves away
                touch_active = False

            # =================================================
            # CONFIRM NEW TOUCH
            # =================================================

            if (
                uzair_close_frames >= MIN_CONSECUTIVE_FRAMES
                and not touch_active
            ):

                # NEW TOUCH EVENT
                uzair_touch_count += 1

                touch_active = True
                likely_touch = True

                current_touch_start_frame = frame_number
                current_touch_start_time = current_time

                # Show message
                touch_message_frames = TOUCH_MESSAGE_DURATION

                print(
                    f"UZAIR TOUCH #{uzair_touch_count} "
                    f"at {current_time:.2f} seconds "
                    f"(Frame {frame_number})"
                )

    # ========================================================
    # DISPLAY MAIN COUNTER
    # ========================================================

    # Background rectangle
    cv2.rectangle(
        frame,
        (20, 20),
        (450, 115),
        (0, 0, 0),
        -1
    )

    # Border
    cv2.rectangle(
        frame,
        (20, 20),
        (450, 115),
        (0, 255, 255),
        2
    )

    # Player name
    cv2.putText(
        frame,
        "PLAYER: UZAIR (ID 13)",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    # Touch counter
    cv2.putText(
        frame,
        f"BALL TOUCHES: {uzair_touch_count}",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # ========================================================
    # TOUCH MESSAGE
    # ========================================================

    if touch_message_frames > 0:

        # Large touch notification
        message_width = 500
        message_height = 90

        message_x = int((width - message_width) / 2)
        message_y = 140

        cv2.rectangle(
            frame,
            (message_x, message_y),
            (message_x + message_width,
             message_y + message_height),
            (0, 0, 0),
            -1
        )

        cv2.rectangle(
            frame,
            (message_x, message_y),
            (message_x + message_width,
             message_y + message_height),
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "UZAIR TOUCH!",
            (message_x + 70, message_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 255),
            3
        )

        touch_message_frames -= 1

    # ========================================================
    # FRAME NUMBER / TIME
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Time: {current_time:.2f}s",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SAVE CSV DATA
    # ========================================================

    results_data.append({
        "Frame": frame_number,
        "Time_Seconds": round(current_time, 3),
        "Player_ID": UZAIR_ID,
        "Player_Name": "Uzair",
        "Ball_Detected": ball_box is not None,
        "Distance": (
            round(ball_distance, 2)
            if ball_distance is not None
            else None
        ),
        "Likely_Touch": likely_touch,
        "Total_Touch_Count": uzair_touch_count
    })

    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Uzair Ball Touch Detection",
        frame
    )

    # Press Q to stop
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("\nStopped by user.")
        break

    # Write frame to output video
    out.write(frame)

# ============================================================
# RELEASE
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results_data)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("UZAIR BALL TOUCH REPORT")
print("========================================")

print(f"Player: Uzair")
print(f"Tracking ID: {UZAIR_ID}")
print(f"Total Separate Touches: {uzair_touch_count}")

print("----------------------------------------")
print(f"Output Video:")
print(OUTPUT_VIDEO)

print("----------------------------------------")
print(f"CSV:")
print(OUTPUT_CSV)

print("========================================")
print("Processing complete!")
