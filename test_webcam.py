from ultralytics import YOLO

# Tải mô hình đã huấn luyện
model = YOLO("best.pt")

print("Đã tải mô hình!")
print("Các lớp:", model.names)

# Nhận diện qua webcam
model.predict(
    source=0,
    show=True,
    conf=0.4,
    imgsz=416,
    device="cpu",
    verbose=False
)
