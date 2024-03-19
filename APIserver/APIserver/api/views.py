# from rest_framework.decorators import api_view
# from rest_framework import status
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
import mimetypes
from rest_framework.parsers import JSONParser 
from django.http import StreamingHttpResponse
from django.http.response import JsonResponse
from django.conf import settings
from rest_framework import status
import requests
import math
import sys
import os
import cv2
import argparse
import cvzone

from tqdm import tqdm
import numpy as np

from ultralytics import YOLO
from APIserver.api.utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, startup_message, download_missing_model_files, get_correct_path
# from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, \
#                   startup_message,download_missing_model_files,get_correct_path

class ShotDetector:
    def __init__(self):
        # Download missing model files required for basketball and rim detection
        download_missing_model_files()
        
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO(get_correct_path(os.path.join("data","models","yolov8s_bb_det_bigdtst_v2_200e_bst.pt")))
        self.class_names = ['Basketball', 'Made','Person','Basketball Hoop','shoot']

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        
        # Debugging
        self.debug = False
        
        self.vid_path = ""

    def process_vid(self,vid_path, disp=False, save_vid=False,debug = False):
        self.debug = debug
        
        self.vid_path = vid_path
        
        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture(vid_path)
        
        # Set the desired width or height
        desired_width = 1280
        desired_height = 720

        # Get the original dimensions
        original_height, original_width = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        # Calculate the aspect ratio (width to height)
        aspect_ratio = original_width / original_height

        # Decide whether to resize based on width or height
        # Compare aspect ratios to determine which dimension to use for scaling
        if desired_width / desired_height > aspect_ratio:
            # The desired aspect ratio is wider than the original, so scale based on height
            new_width = int(desired_height * aspect_ratio)
            new_height = desired_height
        else:
            # The desired aspect ratio is narrower than the original, so scale based on width
            new_width = desired_width
            new_height = int(desired_width / aspect_ratio)

        # Set the new dimensions of the image
        self.img_width, self.img_height = new_width, new_height
        print("vivivivi", vid_path)
        
        self.run(disp, save_vid)
        
    def run(self, disp=False, save_vid=False):
        #goals = 0
        # Initialize video writer if save_vid is True
        if save_vid:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
            # Assuming self.vid_path is the path to your video
            root, ext = os.path.splitext(self.vid_path)
            output_path = root + '_processed.mp4'

            out = cv2.VideoWriter(output_path, fourcc, 20.0, (self.img_width, self.img_height))

        while True:
            ret, self.frame = self.cap.read()       
            if not ret:
                # End of the video or an error occurred
                break

            # Resize the image
            self.frame = cv2.resize(self.frame, (self.img_width, self.img_height))

            # Detect Rim and basketball in frame
            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    pred_cls = int(box.cls[0])
                    if pred_cls not in [0,1,3]:
                        continue

                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and (current_class == "Basketball" or current_class =="shoot"):
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))
                        
                    # # Create hoop points if high confidence
                    # if conf > .25 and current_class == "Made":
                    #     self.hoop_pos.append((center, self.frame_count, w, h, conf))
                    #     cvzone.cornerRect(self.frame, (x1, y1, w, h),colorC=(255,255,255))
                    #     goals+=1
                    #     cv2.imshow(f'Goal_{goals}', self.frame)

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            if disp:
                cv2.imshow("|| Basketball-Shot Counter App ||", self.frame)
                k = cv2.waitKey(1)
                if k == 27:  # Press 'ESC' to exit
                    break
            if save_vid:
                out.write(self.frame)  # Save the frame

        if save_vid:
            out.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)
            
        if self.debug:
            if len(self.ball_pos) > 1:
                # X and Y coordinates
                x1 = self.ball_pos[-2][0][0]
                y1 = self.ball_pos[-2][0][1]
                x2 = self.ball_pos[-1][0][0]
                y2 = self.ball_pos[-1][0][1]
                cv2.line(self.frame, (x1,y1), (x2,y2), (255, 255, 255), 2)
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                w1 = self.ball_pos[-2][2]
                h1 = self.ball_pos[-2][3]
                max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)
                
                # Calculate midpoint for text
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)

                # Frame count
                f1 = self.ball_pos[-2][1]
                f2 = self.ball_pos[-1][1]
                f_dif = f2 - f1

                # Text settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                color = (0, 255, 0)  # Green color
                thickness = 2
                text = f"Dist: {dist:.2f} - f_dif: {f_dif}"

                # Put text on the frame
                cv2.putText(self.frame, text, (mid_x, mid_y), font, scale, color, thickness)
                # if dist >max_dist:
                #     cv2.imshow("Large change",self.frame)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            if len(self.hoop_pos) > 1:
                cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 3 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay
                    if score(self.ball_pos, self.hoop_pos,self.frame,self.attempts,self.debug):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames

                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames
                        
        if self.debug:
            # Display the 'up' and 'down' state on the frame
            up_down_text = f"Up: {self.up}, Down: {self.down}"
            cv2.putText(self.frame, up_down_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # The last recorded position of the hoop is hoop_positions[-1]
            # hoop_positions[(center, self.frame_count, w, h, conf)]
            # hoop_positions[-1][0][1] is the y (row)-coordinate of the hoop's center
            # hoop_positions[-1][3] is the height of the hoop's bounding box
            
            if len(self.hoop_pos)>0:
                hoop_rim_lft_x = int(self.hoop_pos[-1][0][0] - 0.5 * self.hoop_pos[-1][2])
                hoop_rim_top_y = int(self.hoop_pos[-1][0][1] - 0.5 * self.hoop_pos[-1][3])
                hoop_rim_rgt_x = int(self.hoop_pos[-1][0][0] + 0.5 * self.hoop_pos[-1][2])
                hoop_rim_btm_y = int(self.hoop_pos[-1][0][1] + 0.5 * self.hoop_pos[-1][3])

                hoop_rim_toplft = (hoop_rim_lft_x,hoop_rim_top_y)  # y-coordinate of hoop's rim
                hoop_rim_toprgt = (hoop_rim_rgt_x,hoop_rim_top_y)  # y-coordinate of hoop's rim
                hoop_rim_botlft = (hoop_rim_lft_x,hoop_rim_btm_y)  # y-coordinate of hoop's rim
                hoop_rim_botrgt = (hoop_rim_rgt_x,hoop_rim_btm_y)  # y-coordinate of hoop's rim

                # Drawing the trajectory line on the frame
                cv2.line(self.frame, hoop_rim_toplft, hoop_rim_toprgt, (255, 0, 0), 2)
                cv2.putText(self.frame,f"{hoop_rim_toplft}",(hoop_rim_lft_x-15,hoop_rim_top_y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
                cv2.line(self.frame, hoop_rim_botlft, hoop_rim_botrgt, (0, 0, 255), 2)

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def reset(self):
        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

def tutorial(path, disp=False, save_vid=False,debug = False):
        shot_det = ShotDetector()
        if os.path.isdir(path):
            print(f"[INFO]: Processing videos in {path}(same directory as executable)")
            video_files = []
            for filename in os.listdir(path):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):  # Add other video formats if needed
                    vid_path = os.path.join(path, filename)
                    video_files.append(vid_path)        
            for vid_path in tqdm(video_files, desc=f" || Processing {vid_path} ||"):
                print(f"Processing {os.path.basename(vid_path)}")
                shot_det.process_vid(vid_path, disp, save_vid=True)
                shot_det.reset()
        elif os.path.isfile(path):
            vid_path = path
            shot_det.process_vid(vid_path,disp,save_vid,debug)
        else:
            print("[ERROR] Given path is neither Video Nor Directory!")
@api_view(http_method_names=['POST'])
def process(request):
    data = request.POST
    path = data['path']
    # Create obj of Basketball-shot detector class
    # parser = argparse.ArgumentParser(description="Basketball-Shot Counter Application")
    # Make vid_path a positional argument that is optional
    # parser.add_argument("vid_path", nargs='?', type=str, default="", help="Path to the video file")
    # parser.add_argument("--disp", action="store_true", help="Display the processed video")
    # parser.add_argument("--debug", action="store_true", help="Debug App")
    # parser.add_argument('--save', action='store_true')
    # parser.add_argument('--no-save', dest='save', action='store_false')
    # parser.set_defaults(save=True)
    
    # args = parser.parse_args()

    default_path = r"data\demo.mp4"  # Path to your default demo video
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the current directory of the executable
    tutorial(path, save_vid=True)

    # Check if a video path is provided either as a positional argument or by drag-and-drop
    # if args.vid_path or (len(sys.argv) > 1 and sys.argv[1] != '--debug'):
    #     vid_path = args.vid_path if args.vid_path else sys.argv[1]
    #     tutorial(vid_path, args.disp, save_vid=args.save,debug = args.debug)
    # else:
    #     if args.debug:
    #         print(f"[INFO] No specific video file provided! \nChecking for videos in > .\n   [ {current_directory} ]")
    #     if any(file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')) for file in os.listdir(current_directory)):
    #         tutorial(current_directory, disp=args.disp,save_vid=True)
    #     else:
    #         print(startup_message)
    #         default_path_ = get_correct_path(default_path)
    #         print(f"[INFO] No vido file in CWD! \nRunning the default demo video in {default_path_}...")
    #         print("defaultpath:", default_path)
    #         tutorial(default_path_, True, save_vid=False,debug = args.debug)
    return Response({"message": "Processing is success!", "video path": path}, status=status.HTTP_200_OK)

@api_view(http_method_names=["GET"])
def download_file(request, name):

    # fill these variables with real values   vid_id = 10tQGaAD_9c2am4_Yo6qVjf4YMBj5s6Fn
    filename = request.query_params['filename']
    url = f"https://drive.google.com/uc?export=download&id={name}"

    # Make a request to the URL to get the file content
    response = requests.get(url, stream=True)
    
    # Guess the MIME type of the file based on the filename
    mime_type, _ = mimetypes.guess_type(filename)
    
    # Create a Django HTTP response with the content, MIME type, and suggested filename
    django_response = HttpResponse(response.content, content_type=mime_type)
    django_response['Content-Disposition'] = f"attachment; filename={filename}"

    resp = StreamingHttpResponse(
        streaming_content=response.iter_content(chunk_size=1024*1024),
        content_type=response.headers['Content-Type']
    )

    # Add content disposition header to prompt download on the client side
    resp['Content-Disposition'] = f'attachment; filename="{filename}.mp4"'

    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    
    # Write the content to a file in the media directory
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    # Return the path or a message indicating the file has been saved
    return HttpResponse(f"File has been downloaded and saved at: {file_path}") 
