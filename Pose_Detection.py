from ultralytics import YOLO
import os
import pandas
# Load YOLOv11 pose model
model = YOLO("model/yolo11n-pose.pt")


# results = model("D:\\Photos Storage_p\\onkar and me\\IMG-20250121-WA0022.jpg")
# Ask user to paste the path
image_path = input("Enter the path to your image: ").strip('"')



results = model(image_path)


for result in results:
    result.save(filename=os.path.join("output", "output.jpg"))


print("Pose detection completed. Check output.jpg")