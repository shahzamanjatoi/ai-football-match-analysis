import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# SETTINGS
# ============================================================

INPUT_VIDEO = "input/futsal_match.mp4"
OUTPUT_VIDEO = "output/futsal_pass_detection_v2.mp4"
OUTPUT_CSV = "results/futsal_pass_results_v2.csv"

MODEL_PATH = "models/yolo11n.pt"

# Uzair
UZAIR_ID = 13

# YOLO
CONFIDENCE = 0.15
IMAGE_SIZE = 960

# ============================================================
# POSSESSION SETTINGS
# ============================================================

# Ball close to player
POSSESSION_DISTANCE = 60

# Frames needed to establish possession
MIN_POSSESSION_FRAMES = 2

# ============================================================
# PASS SETTINGS
# ============================================================

# Ball must travel at least this many pixels
MIN_PASS_DISTANCE = 50

# Maximum distance between ball and receiving player
RECEIVER_DISTANCE = 70

# Frames receiver should be close to ball
MIN_RECEIVER_FRAMES = 2

# Maximum pass duration
MAX_PASS_FRAMES = 90

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

print()
print("========================================")
print("VIDEO INFORMATION")
print("========================================")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps:.2f}")
print(f"Frames: {total_frames}")
print(f"Duration: {total_frames / fps:.2f} seconds")
print("========================================")

# ============================================================
# OUTPUT
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

frame_number = 0

# ------------------------------------------------------------
# UZAIR POSSESSION
# ------------------------------------------------------------

uzair_close_frames = 0
uzair_has_possession = False

# ------------------------------------------------------------
# BALL HISTORY
# ------------------------------------------------------------

previous_ball_center = None

# ------------------------------------------------------------
# PASS STATE
# ------------------------------------------------------------

pass_in_progress = False

pass_start_frame = None
pass_start_time = None

pass_start_ball = None

receiver_id = None
receiver_close_frames = 0

max_ball_distance = 0

# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

completed_passes = 0

pass_records = []

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

pass_message_frames = 0
PASS_MESSAGE_DURATION = int(fps * 1.0)

last_receiver_id = None

# ============================================================
# HELPER FUNCTION
# ============================================================

def point_to_box_distance(point, box):

    px, py = point

    x1, y1, x2, y2 = box

    nearest_x = max(x1, min(px, x2))
    nearest_y = max(y1, min(py, y2))

    distance = np.sqrt(
        (px - nearest_x) ** 2 +
        (py - nearest_y) ** 2
    )

    return distance


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("Processing video...")
print("Press Q to stop.")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

    # ========================================================
    # YOLO TRACKING
    # ========================================================

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

    # ========================================================
    # READ DETECTIONS
    # ========================================================

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for i in range(len(boxes)):

            cls_id = int(boxes.cls[i])

            confidence = float(boxes.conf[i])

            x1, y1, x2, y2 = map(
                int,
                boxes.xyxy[i].cpu().numpy()
            )

            # PERSON
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

            # SPORTS BALL
            elif cls_id == 32:

                ball_box = (x1, y1, x2, y2)

    # ========================================================
    # BALL CENTER
    # ========================================================

    ball_center = None

    if ball_box is not None:

        bx1, by1, bx2, by2 = ball_box

        ball_center = (
            int((bx1 + bx2) / 2),
            int((by1 + by2) / 2)
        )

        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 0, 255),
            2
        )

        cv2.circle(
            frame,
            ball_center,
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

    # ========================================================
    # FIND UZAIR
    # ========================================================

    uzair_box = None

    for player in player_boxes:

        player_id = player["id"]

        x1, y1, x2, y2 = player["box"]

        if player_id == UZAIR_ID:

            uzair_box = player["box"]

            # Yellow box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "UZair (ID 13)",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
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
    # UZAIR DISTANCE TO BALL
    # ========================================================

    uzair_distance = None

    if ball_center is not None and uzair_box is not None:

        uzair_distance = point_to_box_distance(
            ball_center,
            uzair_box
        )

        ux1, uy1, ux2, uy2 = uzair_box

        nearest_x = max(
            ux1,
            min(ball_center[0], ux2)
        )

        nearest_y = max(
            uy1,
            min(ball_center[1], uy2)
        )

        cv2.line(
            frame,
            ball_center,
            (nearest_x, nearest_y),
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Ball Distance: {uzair_distance:.1f}px",
            (ux1, min(uy2 + 25, height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    # ========================================================
    # IMPORTANT:
    # CHECK FOR PASS START BEFORE RESETTING POSSESSION
    # ========================================================

    if not pass_in_progress:

        # ----------------------------------------------------
        # FIRST: Establish possession
        # ----------------------------------------------------

        if (
            uzair_distance is not None
            and uzair_distance <= POSSESSION_DISTANCE
        ):

            uzair_close_frames += 1

        else:

            uzair_close_frames = 0

        if uzair_close_frames >= MIN_POSSESSION_FRAMES:

            uzair_has_possession = True

        # ----------------------------------------------------
        # SECOND: Ball leaves Uzair
        # ----------------------------------------------------

        if (
            uzair_has_possession
            and uzair_distance is not None
            and uzair_distance > POSSESSION_DISTANCE
            and ball_center is not None
        ):

            # START PASS
            pass_in_progress = True

            pass_start_frame = frame_number

            pass_start_time = current_time

            pass_start_ball = ball_center

            receiver_id = None

            receiver_close_frames = 0

            max_ball_distance = 0

            print(
                f"Possible pass STARTED "
                f"at {current_time:.2f}s"
            )

            # Important:
            # possession is now finished
            uzair_has_possession = False

            uzair_close_frames = 0

    # ========================================================
    # PASS IN PROGRESS
    # ========================================================

    if pass_in_progress:

        pass_frames = frame_number - pass_start_frame

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        if pass_frames > MAX_PASS_FRAMES:

            print(
                f"Pass rejected - timeout "
                f"at {current_time:.2f}s"
            )

            pass_in_progress = False

            receiver_id = None
            receiver_close_frames = 0

            max_ball_distance = 0

        # ----------------------------------------------------
        # BALL TRAVEL DISTANCE
        # ----------------------------------------------------

        if (
            ball_center is not None
            and pass_start_ball is not None
        ):

            ball_travel = np.sqrt(
                (
                    ball_center[0]
                    - pass_start_ball[0]
                ) ** 2
                +
                (
                    ball_center[1]
                    - pass_start_ball[1]
                ) ** 2
            )

            max_ball_distance = max(
                max_ball_distance,
                ball_travel
            )

            # Draw trajectory
            cv2.line(
                frame,
                pass_start_ball,
                ball_center,
                (255, 0, 255),
                2
            )

        # ----------------------------------------------------
        # FIND RECEIVER
        # ----------------------------------------------------

        nearest_player = None
        nearest_distance = float("inf")

        if ball_center is not None:

            for player in player_boxes:

                player_id = player["id"]

                # Never choose Uzair as receiver
                if player_id == UZAIR_ID:
                    continue

                distance = point_to_box_distance(
                    ball_center,
                    player["box"]
                )

                if distance < nearest_distance:

                    nearest_distance = distance

                    nearest_player = player

        # ----------------------------------------------------
        # RECEIVER CLOSE TO BALL
        # ----------------------------------------------------

        if nearest_player is not None:

            candidate_id = nearest_player["id"]

            if nearest_distance <= RECEIVER_DISTANCE:

                if receiver_id == candidate_id:

                    receiver_close_frames += 1

                else:

                    receiver_id = candidate_id

                    receiver_close_frames = 1

                # Highlight possible receiver
                rx1, ry1, rx2, ry2 = (
                    nearest_player["box"]
                )

                cv2.rectangle(
                    frame,
                    (rx1, ry1),
                    (rx2, ry2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"RECEIVER ID {candidate_id}",
                    (rx1, max(ry1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                receiver_close_frames = 0

        # ====================================================
        # COMPLETED PASS
        # ====================================================

        if (
            receiver_id is not None
            and receiver_close_frames >= MIN_RECEIVER_FRAMES
            and max_ball_distance >= MIN_PASS_DISTANCE
        ):

            completed_passes += 1

            pass_end_time = current_time

            # Save result
            pass_records.append({

                "Pass_Number":
                    completed_passes,

                "From_Player":
                    "Uzair",

                "From_ID":
                    UZAIR_ID,

                "To_Player":
                    f"Player {receiver_id}",

                "To_ID":
                    receiver_id,

                "Start_Time":
                    round(pass_start_time, 3),

                "End_Time":
                    round(pass_end_time, 3),

                "Duration":
                    round(
                        pass_end_time
                        - pass_start_time,
                        3
                    ),

                "Ball_Distance":
                    round(
                        max_ball_distance,
                        2
                    )
            })

            last_receiver_id = receiver_id

            # Show message
            pass_message_frames = (
                PASS_MESSAGE_DURATION
            )

            print()
            print("========================================")
            print("PASS COMPLETED!")
            print("========================================")
            print(
                f"Pass Number: "
                f"{completed_passes}"
            )
            print(
                f"Uzair (ID 13) -> "
                f"Player {receiver_id}"
            )
            print(
                f"Time: "
                f"{pass_end_time:.2f}s"
            )
            print(
                f"Ball Travel: "
                f"{max_ball_distance:.1f}px"
            )
            print("========================================")
            print()

            # Reset
            pass_in_progress = False

            receiver_id = None

            receiver_close_frames = 0

            max_ball_distance = 0

    # ========================================================
    # MAIN INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (15, 15),
        (510, 135),
        (0, 0, 0),
        -1
    )

    cv2.rectangle(
        frame,
        (15, 15),
        (510, 135),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "PLAYER: UZAIR (ID 13)",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"COMPLETED PASSES: {completed_passes}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if pass_in_progress:

        status = "PASS IN PROGRESS..."

        status_color = (255, 0, 255)

    elif uzair_has_possession:

        status = "UZAIR HAS BALL"

        status_color = (0, 255, 0)

    else:

        status = "WAITING"

        status_color = (255, 255, 255)

    cv2.putText(
        frame,
        status,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )

    # ========================================================
    # PASS COMPLETED MESSAGE
    # ========================================================

    if pass_message_frames > 0:

        message_width = 650
        message_height = 130

        message_x = int(
            (width - message_width) / 2
        )

        message_y = 150

        cv2.rectangle(
            frame,
            (message_x, message_y),
            (
                message_x + message_width,
                message_y + message_height
            ),
            (0, 0, 0),
            -1
        )

        cv2.rectangle(
            frame,
            (message_x, message_y),
            (
                message_x + message_width,
                message_y + message_height
            ),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "PASS COMPLETED!",
            (
                message_x + 145,
                message_y + 50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Uzair -> Player {last_receiver_id}",
            (
                message_x + 125,
                message_y + 100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        pass_message_frames -= 1

    # ========================================================
    # FRAME / TIME
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
    # SAVE / DISPLAY
    # ========================================================

    out.write(frame)

    cv2.imshow(
        "Uzair Pass Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\nStopped by user.")

        break

# ============================================================
# RELEASE
# ============================================================

cap.release()
out.release()

cv2.destroyAllWindows()

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(pass_records)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("========================================")
print("UZAIR PASS DETECTION REPORT")
print("========================================")

print("Player: Uzair")
print("Tracking ID: 13")

print(
    f"COMPLETED PASSES: "
    f"{completed_passes}"
)

print("----------------------------------------")

if completed_passes > 0:

    for record in pass_records:

        print(
            f"Pass {record['Pass_Number']}: "
            f"Uzair -> "
            f"{record['To_Player']} | "
            f"Time: "
            f"{record['End_Time']}s"
        )

else:

    print("No completed passes detected.")

print("----------------------------------------")

print(
    f"Output Video:"
)

print(
    OUTPUT_VIDEO
)

print("----------------------------------------")

print(
    f"CSV Report:"
)

print(
    OUTPUT_CSV
)

print("========================================")
print("Processing complete!")
