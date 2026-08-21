import os
from ultralytics import YOLO

# Paths
model_path = "runs/rocket_yolov82/weights/best.pt"
media_folder = "testing_media"
output_folder = "output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the model
model = YOLO(model_path)

# Supported extensions
image_exts = [".jpg", ".jpeg", ".png"]
video_exts = [".mp4", ".avi", ".mov", ".mkv"]

# Loop through files in media folder
for file_name in os.listdir(media_folder):
    file_path = os.path.join(media_folder, file_name)
    ext = os.path.splitext(file_name)[1].lower()

    if ext in image_exts + video_exts:
        print(f"Processing {file_name} ...")

        # Set save=True and save_dir=output_folder to save directly in output
        results = model(file_path, save=True, save_dir=output_folder)

print(f"All files processed! Check the '{output_folder}' folder for annotated outputs.")
