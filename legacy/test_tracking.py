from ultralytics import YOLO
import sys
import os

def test_tracking(video_path, model_path='runs/train/rocket_detector/weights/best.pt'):
    """
    Test the trained model on a video with tracking
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model weights
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train the model first using: python train.py")
        return
    
    print(f"Loading model from: {model_path}")
    print(f"Processing video: {video_path}")
    
    model = YOLO(model_path)
    
    # Check ultralytics version
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except:
        pass
    
    try:
        # Run tracking with default settings
        results = model.track(
            source=video_path,
            conf=0.5,
            iou=0.5,
            persist=True,
            show=False,
            save=True,
            save_txt=True,
            line_width=2,
            vid_stride=1,
            stream=True,
            verbose=True,
            imgsz=640,
            device=0,
        )
        
        # Process results
        total_frames = 0
        frames_with_detections = 0
        
        print("\nProcessing video...")
        for i, r in enumerate(results):
            total_frames += 1
            
            boxes = r.boxes
            if len(boxes) > 0:
                frames_with_detections += 1
                
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()
                    
                    print(f"Frame {i}: {len(track_ids)} rocket(s) - IDs: {track_ids.tolist()}, Conf: {confidences.round(2).tolist()}")
        
        print("\n" + "="*50)
        print("Tracking Complete!")
        print("="*50)
        print(f"Total frames: {total_frames}")
        print(f"Frames with detections: {frames_with_detections}")
        if total_frames > 0:
            print(f"Detection rate: {frames_with_detections/total_frames*100:.1f}%")
        
        # Find output
        output_dir = "runs/track"
        if os.path.exists(output_dir):
            latest_exp = sorted([d for d in os.listdir(output_dir) if d.startswith('predict')])
            if latest_exp:
                output_path = os.path.join(output_dir, latest_exp[-1])
                print(f"\nOutput: {output_path}")
    
    except AttributeError as e:
        if 'fuse_score' in str(e):
            print("\n" + "="*50)
            print("ERROR: Outdated Ultralytics Version")
            print("="*50)
            print("\nYour ultralytics package is outdated. Fix with:")
            print("\n  pip uninstall ultralytics -y")
            print("  pip install ultralytics")
            print("\nOr try detection without tracking:")
            print(f"  python detect_only.py {video_path}")
        else:
            raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tracking.py <video_path> [model_path]")
        print("\nExamples:")
        print("  python test_tracking.py testvid.mp4")
        print("  python test_tracking.py testvid.mp4 runs/train/rocket_yolov82/weights/best.pt")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'runs/train/rocket_detector/weights/best.pt'
    
    test_tracking(video_path, model_path)

if __name__ == "__main__":
    main()