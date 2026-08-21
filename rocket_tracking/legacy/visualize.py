import os
import random
import cv2

# Splits
splits = ["train", "valid", "test"]

save_dir = "visualized"
os.makedirs(save_dir, exist_ok=True)

# Mapping class index to names
class_names = {0: "rocket"}  # change if you have more classes

# Function to draw bounding boxes on an image
def draw_boxes(img_path, lbl_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, class_names[int(cls)], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

# Visualize a few images per split
for split in splits:
    img_dir = os.path.join(split, "images")
    lbl_dir = os.path.join(split, "labels")

    imgs = os.listdir(img_dir)
    sample_imgs = random.sample(imgs, min(5, len(imgs)))  # pick 5 random images

    for img_name in sample_imgs:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))
        out_img = draw_boxes(img_path, lbl_path)
        cv2.imwrite(os.path.join(save_dir, f"{split}_{img_name}"), out_img)

print(f"Visualization saved in {save_dir}")
