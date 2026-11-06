
from ultralytics import YOLO
import os

model_path = "best.pt"

print("File mo hinh ton tai:", os.path.exists(model_path))

if os.path.exists(model_path):
    print("Dung luong file:", os.path.getsize(model_path))

    model = YOLO(model_path)

    print("\nDanh sach lop:")
    print(model.names)

    print("\nSo lop:", len(model.names))
else:
    print("KHONG TIM THAY FILE best.pt")
