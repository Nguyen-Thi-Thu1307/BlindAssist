import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam")
    exit()

print("Nhan S de luu anh webcam")
print("Nhan ESC de thoat")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.1,
        imgsz=640,
        verbose=False
    )

    display_frame = results[0].plot()
    cv2.imshow("YOLO26n Webcam", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("webcam_test.jpg", frame)
        print("Da luu anh webcam_test.jpg")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
