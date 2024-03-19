import math
import numpy as np
import cv2
import os
import gdown
import subprocess
import sys

# Define the startup message at the beginning of your script
startup_message = '''
================================================================================
                   Welcome to the Basketball-Shot Counter Application!
================================================================================

This application uses advanced computer vision techniques to count basketball-shots in a video. You can run it in various modes depending on your needs.

Usage:

1. Process a Specific Video File (Drag-&-Drop-to-exe) || (Cmd):
   Command: python bbshot_counter_app.py <path_to_video>
   Example: python bbshot_counter_app.py videos/basketball_match.mp4

2. Process All Videos in the Current Directory:
   Command: python bbshot_counter_app.py
   This mode will automatically find and process all video files in the same directory as the executable.

Arguments:

- vid_path: (Optional) Path to a specific video file.
- --disp: (Optional) Display the processed video in a window.
- --debug: (Optional) Run the application in debug mode for additional output.
- --save [True|False]: (Optional) Choose whether to save the processed video. Default is True.

Examples:

- To run with a specific video and display the output: 
  python bbshot_counter_app.py myvideo.mp4 --disp

- To process videos in the current directory without displaying them:
  python bbshot_counter_app.py --save True

- To run in debug mode with a specific video:
  python bbshot_counter_app.py myvideo.mp4 --debug

Demo Mode:

Running the application without any specific video will trigger the demo mode. In this mode, the application will:
- Check for videos in the current directory.
- If no videos are found, it will run a default demo video.

================================================================================
'''



def get_correct_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def score(ball_traj, hoop_data, frame, attempt_num,debug):
    disp_attempts = False
    
    draw_frame = frame.copy()
    traj_x = []
    traj_y = []

    hoop_rim_top_y = hoop_data[-1][0][1] - 0.5 * hoop_data[-1][3]  # y-coordinate of hoop's rim top
    hoop_rim_btm_y = hoop_data[-1][0][1] + 0.5 * hoop_data[-1][3]  # y-coordinate of hoop's rim bottom
    hoop_rim_center_y = hoop_data[-1][0][1]

    point_below_rim = None
    point_above_rim = None

    # Loop over the ball trajectory in reverse
    for i in reversed(range(len(ball_traj))):
        current_y = ball_traj[i][0][1]

        if current_y > hoop_rim_btm_y and point_below_rim is None:
            if i > 0:  # Ensure there is a previous point
                point_below_rim = ball_traj[i - 1][0]

        if current_y < hoop_rim_top_y and point_above_rim is None:
            point_above_rim = ball_traj[i][0]
            break

    # Proceed if both points are found
    if point_below_rim and point_above_rim:
        traj_x.extend([point_above_rim[0], point_below_rim[0]])
        traj_y.extend([point_above_rim[1], point_below_rim[1]])
        
        # Create line from two points and display the image
        if len(traj_x) > 1:
            if traj_x[0] == traj_x[1]:  # Check for vertical line
                # Handle the vertical line case
                # For example, you might just use the x-coordinate to define the line
                predicted_x = traj_x[0]
            else:
                m, b = np.polyfit(traj_x, traj_y, 1)
                # Check if projected line fits between the ends of the rim
                predicted_x = ((hoop_rim_center_y - b) / m)
                
            rim_x1 = hoop_data[-1][0][0] - 0.4 * hoop_data[-1][2]
            rim_x2 = hoop_data[-1][0][0] + 0.4 * hoop_data[-1][2]

            if debug:
                # Drawing the line on the frame
                cv2.line(draw_frame, (traj_x[0], traj_y[0]), (traj_x[1], traj_y[1]), (0, 255, 0), 2)  

                cv2.circle(draw_frame,(int(predicted_x),int(hoop_rim_center_y)),3,(0,0,255),-1)

                if disp_attempts:
                    # Displaying the frame with the line
                    cv2.imshow(f'Attempt-Made: {attempt_num}', draw_frame)
                    #cv2.waitKey(0)


            if rim_x1 < predicted_x < rim_x2:
                return True



    return False


# Detects if the ball is below the net - used to detect shot attempts
def detect_down(ball_pos, hoop_pos):
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    if ball_pos[-1][0][1] > y:
        return True
    return False


# Detects if the ball is around the backboard - used to detect shot attempts
def detect_up(ball_pos, hoop_pos):
    curr_ball_pos = ball_pos[-1][0]
    x1 = hoop_pos[-1][0][0] - 4 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 4 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 4 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]

    if x1 < curr_ball_pos[0] < x2 and y1 < curr_ball_pos[1] < y2:
        return True
    return False


# Checks if center point is near the hoop
def in_hoop_region(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


# Removes inaccurate data points
def clean_ball_pos(ball_pos, frame_count):
    # Removes inaccurate ball size to prevent jumping to wrong ball
    if len(ball_pos) > 1:
        # Width and Height
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X and Y coordinates
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # Frame count
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 2.5 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move a 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 10):
            ball_pos.pop()

        # Ball should be relatively square [1 side cannot be too greater then the other as ball is round]
        elif (w2*1.8 < h2) or (h2*1.8 < w2):
            ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 50:
            ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    # Prevents jumping from one hoop to another
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2*3 < h2) or (h2*3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 45:
        hoop_pos.pop(0)

    return hoop_pos


def download_missing_model_files():
    """Download missing model files from Google Drive."""
    models_dir = get_correct_path(os.path.join('data', 'models'))
    model_files = ['yolov8s_bb_det_bigdtst_v2_200e_bst.pt']  # replace with actual file names
    files_id    = ['1mhjPDkW0ZuGsm8psPq6pJpUqav6l7OIC']  # replace with actual file IDs or URLs
    
    print("models_dir = ",models_dir)
    # Create model directory if it doesnot exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for i, file in enumerate(model_files):
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            print(f'{file} not found. Downloading...')
            file_id = files_id[i]  # replace with the actual file ID or URL
            if file_id.startswith('http'):
                # Use curl to download the file
                subprocess.run(['curl', '-L', file_id, '-o', file_path], check=True)
            else:
                # Use gdown to download the file from Google Drive
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, file_path, quiet=False)
            print(f'{file} downloaded successfully!')