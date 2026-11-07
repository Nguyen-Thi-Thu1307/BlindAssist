from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO("best.pt")

# Kiểm tra 9 class
print("Danh sách class:")
print(model.names)

# Kiểm tra mô hình
print("\nMô hình đã tải thành công!")