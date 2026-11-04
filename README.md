# AI-Based Football Match Analysis and Player Performance Tracking Using Computer Vision

A computer vision project that uses YOLO object detection and tracking to analyze football/futsal match videos and generate player and match statistics.

## Features

- Player detection and tracking
- Player statistics and frame-by-frame analysis
- Ball detection
- Player ball-touch analysis
- Pass detection
- Data visualization and statistical graphs

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Project Structure

```text
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
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/shahzamanjatoi/ai-football-match-analysis.git
cd ai-football-match-analysis
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the project root directory:

```bash
python src/analysis.py
python src/ball_touch.py
python src/check_tracking_ids.py
python src/pass_detection.py
python src/visualize_results.py
```

## Notes

- Place the input video at `input/futsal_match.mp4`.
- The YOLO model is expected at `models/yolo11n.pt`.
- Generated videos and analysis results are saved in the `output/` and `results/` folders.
- Pass detection is experimental and may require further refinement.

## Author

**Shahzaman Jatoi**

Mechatronics Engineering Student  
Mehran University of Engineering and Technology (MUET)
