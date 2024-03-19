# 🏀 Basketball-Shot Counter Application 🏀

## Description
This project is a sophisticated basketball shot detection and analysis tool using computer vision. It leverages the power of YOLO (You Only Look Once) for object detection and OpenCV for image processing, enabling users to accurately count shots made and attempted in a basketball game from video footage.

## Features
- Shot detection using YOLO and OpenCV
- Real-time processing of video footage
- Counting and displaying made shots and attempts
- Debugging mode for development and testing
- Option to save processed video with annotations

Your README's installation section is quite clear and straightforward, but it can be enhanced for better readability and clarity. Here's a revised version:

## Installation

Follow these steps to get started with the Basketball Shot Counter project. You can choose between GPU or CPU processing based on your system capabilities.

### Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/HaiderAbasi/OpenCV_Basketball_Shot_Counter
cd OpenCV_Basketball_Shot_Counter
```

### Install Dependencies
Depending on your processing preference, install the required dependencies.

#### For GPU Processing (Faster)
If you have a GPU that supports CUDA, you can utilize GPU processing for faster performance:
```bash
pip install -r requirements.txt
```

#### For CPU Processing (Slower)
For systems without GPU support, use CPU processing:
```bash
pip install -r requirements-cpu.txt
```


## Usage
Run the application using the following command:
```bash
python src/bbshot_counter_app.py [path to video file] [--disp] [--debug] [--save/--no-save]
```

- `vid_path`: Path to the video file or directory containing multiple videos.
- `--disp`: Display the processed video in real-time.
- `--debug`: Enable debug mode for additional output.
- `--save`: Save the processed video. Enabled by default.
- `--no-save`: Disable saving the processed video.

## Example
Process a single video and display the output:
```bash
python src/bbshot_counter_app.py ./path/to/video.mp4 --disp
```

Process all videos in a directory without displaying:
```bash
python src/bbshot_counter_app.py ./path/to/videos/
```

Enable debugging for a specific video:
```bash
python src/bbshot_counter_app.py ./path/to/video.mp4 --debug
```

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License
- MIT-License

---