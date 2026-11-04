# Futsal Match Analysis Using YOLO

A computer vision project that uses YOLO object detection and tracking to analyze a futsal match video.

## Features

- Player detection and tracking
- Player statistics and frame-by-frame analysis
- Ball detection
- Player ball-touch analysis
- Pass detection
- Data visualization and statistical graphs

## Technologies Used

- Python
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Project Structure

    football_object_detection/
    ├── input/
    ├── models/
    ├── output/
    ├── results/
    ├── src/
    │   ├── analysis.py
    │   ├── ball_touch.py
    │   ├── check_tracking_ids.py
    │   ├── pass_detection.py
    │   └── visualize_results.py
    ├── requirements.txt
    └── README.md

## Installation

1. Clone the repository:

       git clone https://github.com/shahzamanjatoi/football_object_detection.git
       cd football_object_detection

2. Create a virtual environment:

       python -m venv venv

3. Activate it on Windows:

       .\venv\Scripts\Activate.ps1

4. Install dependencies:

       pip install -r requirements.txt

## Usage

Run the scripts from the project root directory:

    python src/analysis.py
    python src/ball_touch.py
    python src/check_tracking_ids.py
    python src/pass_detection.py
    python src/visualize_results.py

## Note

Place the input video at `input/futsal_match.mp4`.

The YOLO model is expected at `models/yolo11n.pt`.

Generated videos and analysis results are saved in the `output/` and `results/` folders.

## Author

Shahzaman Jatoi

Mechatronics Engineering Student
Mehran University of Engineering and Technology (MUET)
