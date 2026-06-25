# Player Tracking with YOLOv8


import os
import subprocess
import sys
import importlib
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
import zipfile
import warnings
import platform
import time


def ensure_package(package_name, import_name=None):
    module_name = import_name or package_name.replace('-', '_')
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        importlib.import_module(module_name)


for package, module in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
]:
    ensure_package(package, module)

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from ultralytics import YOLO

warnings.filterwarnings('ignore')


class SportsPlayerTracker:
    def __init__(self, model_size='m', conf_threshold=0.3):

        print("=" * 70)
        print("DS5216: REAL Player Tracking + Keypoint Detection")
        print("Using YOLOv8 Pose Model (Pre-trained)")
        print("=" * 70)

        # Create directories
        self.create_directories()

        # Configuration
        self.model_size = model_size
        self.conf_threshold = conf_threshold

        # Load REAL working model
        self.load_model()

        # Colors for different players
        self.player_colors = [
            (0, 255, 0),  # Green - Player 0
            (255, 0, 0),  # Blue - Player 1
            (0, 0, 255),  # Red - Player 2
            (255, 255, 0),  # Cyan - Player 3
            (255, 0, 255),  # Magenta - Player 4
            (0, 255, 255),  # Yellow - Player 5
            (255, 165, 0),  # Orange - Player 6
            (128, 0, 128),  # Purple - Player 7
        ]

        # Keypoint connections (COCO skeleton - 17 points)
        self.skeleton = [
            # Head connections
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Body connections
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            # Torso connections
            (5, 11), (6, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (11, 12)
        ]

        # Keypoint names
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # Results storage
        self.results = []
        self.processing_stats = []

    def create_directories(self):
        directories = ['videos', 'outputs', 'temp']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f" Created: {directory}/")

    def load_model(self):
        print(f"\n🤖 Loading YOLOv8{self.model_size}-pose model...")

        try:
            # This is the REAL model that actually works
            self.model = YOLO(f'yolov8{self.model_size}-pose.pt')

            # Check if model has pose capability
            print(f"  YOLOv8{self.model_size}-pose model loaded successfully!")
            print(f"   Model type: Pose Estimation")
            print(f"   Keypoints: 17 body joints")
            print(f"   Confidence threshold: {self.conf_threshold}")

            # Test with a dummy image to verify
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_result = self.model(test_img, verbose=False)
            print(f"   Model test: PASSED")

        except Exception as e:
            print(f" Error loading model: {e}")
            print("\n Downloading model automatically...")
            self.download_model()

    def download_model(self):
        try:
            print("Downloading YOLOv8 pose model...")
            # Use Ultralytics to auto-download
            self.model = YOLO(f'yolov8{self.model_size}-pose.pt')
            print(" Model downloaded successfully!")
        except Exception as e:
            print(f" Failed to download model: {e}")
            print("\n Please install required packages:")
            print("   pip install ultralytics opencv-python numpy")
            print("\nOr download manually from:")
            print("   https://github.com/ultralytics/ultralytics")
            exit(1)

    def detect_players_and_keypoints(self, frame):
        try:
            # Run YOLOv8 pose estimation
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

            detections = []

            if results.keypoints is not None and results.keypoints.xy is not None:
                # Get boxes and keypoints
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    keypoints = results.keypoints.xy.cpu().numpy()
                    keypoint_scores = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

                    for i in range(len(boxes)):
                        if scores[i] > self.conf_threshold:
                            bbox = boxes[i].astype(int)
                            person_keypoints = keypoints[i]

                            # Process keypoints
                            processed_keypoints = []
                            valid_kpt_count = 0

                            for kpt_idx in range(len(person_keypoints)):
                                x, y = person_keypoints[kpt_idx]

                                # Get confidence score
                                confidence = 1.0
                                if keypoint_scores is not None and i < len(keypoint_scores):
                                    if kpt_idx < len(keypoint_scores[i]):
                                        confidence = float(keypoint_scores[i][kpt_idx])

                                if x > 0 and y > 0 and confidence > 0.1:
                                    valid_kpt_count += 1
                                    processed_keypoints.append({
                                        'x': float(x),
                                        'y': float(y),
                                        'confidence': confidence,
                                        'visible': True
                                    })
                                else:
                                    processed_keypoints.append({
                                        'x': 0.0,
                                        'y': 0.0,
                                        'confidence': 0.0,
                                        'visible': False
                                    })

                            detections.append({
                                'bbox': bbox.tolist(),
                                'score': float(scores[i]),
                                'keypoints': processed_keypoints,
                                'valid_keypoints': valid_kpt_count,
                                'id': i
                            })

            return detections

        except Exception as e:
            print(f" Detection error: {e}")
            return []

    def track_players_in_video(self, video_path, output_name):

        print(f"\n Processing: {os.path.basename(video_path)}")
        print("   Detecting: Players + 17 Body Keypoints + Skeleton")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Cannot open video: {video_path}")
            return 0, None, 0

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output video
        output_path = os.path.join('outputs', f'{output_name}_tracking.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Tracking variables
        tracks = {}  # track_id -> {'positions': [], 'color': (), 'keypoints': []}
        next_track_id = 0
        track_colors = {}

        print(f"   Video: {width}x{height} @ {fps} FPS")
        print(f"   Total frames: {total_frames}")
        print(f"   Processing...")

        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

        frame_count = 0
        total_detections = 0
        total_keypoints = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make a copy for drawing
            display_frame = frame.copy()


            detections = self.detect_players_and_keypoints(frame)
            total_detections += len(detections)

            used_tracks = set()

            for detection in detections:
                bbox = detection['bbox']
                keypoints = detection['keypoints']
                score = detection['score']

                # Calculate center of bbox for tracking
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Find closest existing track
                best_track_id = None
                min_distance = 50  # pixels

                for track_id, track_data in tracks.items():
                    if track_id in used_tracks:
                        continue

                    if track_data['positions']:
                        last_pos = track_data['positions'][-1]
                        distance = np.sqrt((center_x - last_pos[0]) ** 2 +
                                           (center_y - last_pos[1]) ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            best_track_id = track_id

                # Assign or create track
                if best_track_id is not None and min_distance < 50:
                    track_id = best_track_id
                    tracks[track_id]['positions'].append((center_x, center_y))
                    tracks[track_id]['keypoints'] = keypoints
                    used_tracks.add(track_id)
                else:
                    track_id = next_track_id
                    tracks[track_id] = {
                        'positions': [(center_x, center_y)],
                        'keypoints': keypoints,
                        'first_seen': frame_count
                    }
                    track_colors[track_id] = self.player_colors[track_id % len(self.player_colors)]
                    next_track_id += 1
                    used_tracks.add(track_id)

                color = track_colors[track_id]

                # 1. Draw BOUNDING BOX
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # 2. Draw PLAYER ID
                cv2.putText(display_frame, f"Player {track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 3. Draw CONFIDENCE
                cv2.putText(display_frame, f"{score:.2f}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 4. Draw KEYPOINTS (17 body joints)
                for kpt_idx, kpt_data in enumerate(keypoints):
                    if kpt_data['visible']:
                        x = int(kpt_data['x'])
                        y = int(kpt_data['y'])
                        conf = kpt_data['confidence']

                        # Color code by body part
                        if kpt_idx == 0:  # Nose
                            kpt_color = (0, 255, 0)  # Green
                            size = 5
                        elif kpt_idx in [5, 6]:  # Shoulders
                            kpt_color = (0, 255, 255)  # Yellow
                            size = 6
                        elif kpt_idx in [7, 8]:  # Elbows
                            kpt_color = (255, 255, 0)  # Cyan
                            size = 5
                        elif kpt_idx in [9, 10]:  # Wrists
                            kpt_color = (255, 0, 255)  # Magenta
                            size = 4
                        elif kpt_idx in [11, 12]:  # Hips
                            kpt_color = (0, 165, 255)  # Orange
                            size = 6
                        elif kpt_idx in [13, 14]:  # Knees
                            kpt_color = (255, 0, 0)  # Blue
                            size = 5
                        elif kpt_idx in [15, 16]:  # Ankles
                            kpt_color = (0, 0, 255)  # Red
                            size = 4
                        else:  # Eyes, ears
                            kpt_color = (255, 255, 255)  # White
                            size = 3

                        # Draw keypoint
                        cv2.circle(display_frame, (x, y), size, kpt_color, -1)
                        total_keypoints += 1

                # 5. Draw SKELETON CONNECTIONS
                for connection in self.skeleton:
                    start_idx, end_idx = connection

                    if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                        start_kpt = keypoints[start_idx]
                        end_kpt = keypoints[end_idx]

                        if start_kpt['visible'] and end_kpt['visible']:
                            x1_kpt = int(start_kpt['x'])
                            y1_kpt = int(start_kpt['y'])
                            x2_kpt = int(end_kpt['x'])
                            y2_kpt = int(end_kpt['y'])

                            # Draw skeleton line
                            cv2.line(display_frame,
                                     (x1_kpt, y1_kpt),
                                     (x2_kpt, y2_kpt),
                                     color, 2)

            # Draw frame information
            info_y = 30
            cv2.putText(display_frame, f"Frame: {frame_count}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Players: {len(tracks)}",
                        (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame to output video
            out.write(display_frame)
            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()

        # Calculate processing statistics
        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0

        print(f"\n Processing complete!")
        print(f"   Time: {processing_time:.1f} seconds")
        print(f"   Speed: {avg_fps:.1f} FPS")
        print(f"   Players detected: {next_track_id}")
        print(f"   Total detections: {total_detections}")
        print(f"   Total keypoints: {total_keypoints}")
        print(f"   Output saved: {output_path}")

        return next_track_id, output_path, avg_fps

    def select_videos(self):
        """Open file dialog to select videos"""
        print("\n STEP 1: SELECT SPORTS VIDEOS")
        print("-" * 40)

        root = tk.Tk()
        root.withdraw()

        video_files = filedialog.askopenfilenames(
            title="Select Sports Videos (Football, Basketball, etc.)",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )

        if not video_files:
            print(" No videos selected!")
            return []

        video_files = list(video_files)
        print(f"\n Selected {len(video_files)} videos:")
        for i, vpath in enumerate(video_files):
            print(f"   {i + 1}. {os.path.basename(vpath)}")

        return video_files

    def copy_videos(self, video_paths):
        print("\n Copying videos...")
        for video_path in video_paths:
            try:
                filename = os.path.basename(video_path)
                dest_path = os.path.join('videos', filename)

                if not os.path.exists(dest_path):
                    shutil.copy2(video_path, dest_path)
                    print(f"    {filename}")
                else:
                    print(f"  Already exists: {filename}")
            except Exception as e:
                print(f"    Failed to copy {filename}: {e}")

    def package_results(self):

        print("\n STEP 4: PACKAGING RESULTS")
        print("-" * 40)

        zip_path = 'player_tracking_results.zip'

        print("Creating zip file with processed videos...")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all output files
            for root, dirs, files in os.walk('outputs'):
                for file in files:
                    if file.endswith('.mp4'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname)
                        print(f"   Added: {file}")

        print(f"\n Zip file created: {zip_path}")
        return zip_path

    def show_summary(self):

        print("\n" + "=" * 70)
        print(" PROCESSING COMPLETE!")
        print("=" * 70)

        if not self.results:
            print(" No videos were processed")
            return

        total_players = 0
        total_fps = 0

        print("\n DETAILED RESULTS:")
        for result in self.results:
            print(f"\n   Video: {result['video_name']}")
            print(f"     • Players tracked: {result['player_count']}")
            print(f"     • Processing speed: {result['fps']:.1f} FPS")
            print(f"     • Output video: {os.path.basename(result['tracking_path'])}")

            total_players += result['player_count']
            total_fps += result['fps']

        if self.results:
            avg_fps = total_fps / len(self.results)
            print(f"\n OVERALL STATISTICS:")
            print(f"   • Total videos processed: {len(self.results)}")
            print(f"   • Total players detected: {total_players}")
            print(f"   • Average speed: {avg_fps:.1f} FPS")

        print(f"\n WHAT YOU'LL SEE IN VIDEOS:")
        print(f"   1. Player bounding boxes (colored by player)")
        print(f"   2. Player IDs (Player 0, Player 1, etc.)")
        print(f"   3. 17 body keypoints (colored by body part)")
        print(f"   4. Skeleton connections")
        print(f"   5. Confidence scores")
        print(f"   6. NO movement trails (clean output)")

        print(f"\n FOR YOUR ASSIGNMENT:")
        print(f"   1. Open the tracking videos in any video player")
        print(f"   2. Play the videos and take screenshots while playing")
        print(f"   3. Screenshots should show:")
        print(f"      - Player bounding boxes")
        print(f"      - Player IDs")
        print(f"      - Body keypoints (17 joints)")
        print(f"      - Skeleton connections")

    def open_outputs_folder(self):

        response = messagebox.askyesno(
            "Open Folder",
            "Open outputs folder to see your processed videos?"
        )

        if response:
            try:
                outputs_path = os.path.abspath('outputs')
                if platform.system() == "Windows":
                    os.startfile(outputs_path)
                elif platform.system() == "Darwin":
                    os.system(f'open "{outputs_path}"')
                else:
                    os.system(f'xdg-open "{outputs_path}"')
            except:
                print(f" Please open manually: {outputs_path}")


def main():

    print("=" * 70)
    print("DS5216: Player Tracking with REAL Detection")
    print("Using YOLOv8 Pose Model - 17 Body Keypoints")
    print("=" * 70)

    # Check for required packages
    print("\n🔍 Checking required packages...")
    try:
        import ultralytics
        print(f" ultralytics: {ultralytics.__version__}")
    except ImportError:
        print(" ultralytics not installed!")
        print("   Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

    try:
        import torch
        print(f" torch: {torch.__version__}")
    except ImportError:
        print(" torch not installed!")
        print("   Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])

    # Initialize tracker
    print("\n Initializing Player Tracker...")
    tracker = SportsPlayerTracker(
        model_size='m',  # 'n'=nano, 's'=small, 'm'=medium (recommended), 'l'=large, 'x'=xlarge
        conf_threshold=0.25  # Lower = more detections, Higher = fewer but more confident
    )

    # Step 1: Select videos
    video_paths = tracker.select_videos()
    if not video_paths:
        print(" No videos selected. Exiting...")
        return

    print(f"\n {len(video_paths)} videos selected")

    # Step 2: Copy videos
    tracker.copy_videos(video_paths)

    # Step 3: Clear old outputs
    print("\n Clearing old output files...")
    for file in os.listdir('outputs'):
        try:
            os.remove(os.path.join('outputs', file))
        except:
            pass

    # Step 4: Process each video
    print("\n STEP 3: PROCESSING VIDEOS")
    print("=" * 60)

    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path).split('.')[0]

        print(f"\n{'=' * 50}")
        print(f"🎥 Processing Video {i + 1}/{len(video_paths)}: {video_name}")
        print('=' * 50)

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.1f}")
        print(f"   Frames: {total_frames}")

        # Process video
        player_count, tracking_path, fps_speed = tracker.track_players_in_video(
            video_path, video_name
        )

        # Store results
        tracker.results.append({
            'video_name': video_name,
            'player_count': player_count,
            'fps': fps_speed,
            'tracking_path': tracking_path
        })

        print(f"\n    Video {i + 1} Complete")
        print(f"      Players: {player_count}")
        print(f"      Speed: {fps_speed:.1f} FPS")

    # Step 5: Package results
    zip_path = tracker.package_results()

    # Step 6: Show summary
    tracker.show_summary()

    # Final instructions
    print(f"\n YOUR FILES ARE READY:")
    print(f"   1. Zip file: {os.path.abspath(zip_path)}")
    print(f"   2. Output folder: {os.path.abspath('outputs')}")

    print(f"\n Files in outputs folder:")
    output_files = os.listdir('outputs')
    for file in sorted(output_files):
        if file.endswith('.mp4'):
            file_size = os.path.getsize(os.path.join('outputs', file)) / (1024 * 1024)
            print(f"   • {file} ({file_size:.1f} MB)")

    # Open folder
    tracker.open_outputs_folder()

    print("\n" + "=" * 70)
    print(" ALL DONE! Check the 'outputs' folder.")
    print("=" * 70)


if __name__ == "__main__":
    # Run main program
    main()