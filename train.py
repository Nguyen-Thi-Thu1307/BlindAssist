from ultralytics import YOLO

# Load mô hình YOLO26n
model = YOLO("yolo26n.pt")

# Train thử trên CPU
results = model.train(
    data="combined_dataset_9classes/data.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    workers=2,
    device="cpu",
    project="runs/detect",
    name="yolo26n_9classes",
    save=True,
    plots=True
)

print("ĐÃ HOÀN THÀNH TRAIN THỬ!")