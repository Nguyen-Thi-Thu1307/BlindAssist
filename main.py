import cv2
import asyncio
import edge_tts
import pygame
import threading
import queue
import time

from ultralytics import YOLO

# 1. Tai mo hinh YOLO26n
model = YOLO("best.pt")

# 2. Khoi tao hang doi giong noi
speech_queue = queue.Queue()

# Ten vat the tieng Viet
vietnamese_names = {
    "projector": "máy chiếu",
    "desktop_computer": "máy tính để bàn",
    "laptop": "laptop",
    "cell_phone": "điện thoại"
}

# 3. Ham tao va phat giong noi
def speech_worker():
    pygame.mixer.init()

    while True:
        message = speech_queue.get()

        if message is None:
            speech_queue.task_done()
            break

        try:
            async def create_voice():
                communicate = edge_tts.Communicate(
                    message,
                    voice="vi-VN-HoaiMyNeural"
                )
                await communicate.save("speech.mp3")

            asyncio.run(create_voice())

            pygame.mixer.music.load("speech.mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print("Loi TTS:", e)

        finally:
            speech_queue.task_done()

    pygame.mixer.quit()


# Chay TTS tren luong rieng
speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)
speech_thread.start()

# 4. Mo webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam!")
    speech_queue.put(None)
    speech_thread.join()
    exit()

print("He thong dang hoat dong.")
print("Nhan ESC de thoat.")

last_message = ""
last_speech_time = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        height, width = frame.shape[:2]

        # Nhan dien vat the
        results = model.predict(
            source=frame,
            conf=0.45,
            imgsz=640,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) > 0:

            # Lay vat the co confidence cao nhat
            box = max(
                boxes,
                key=lambda b: float(b.conf[0])
            )

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]
            object_name = vietnamese_names.get(
                class_name,
                class_name
            )

            # Lay toa do vat the
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2

            # Xac dinh vi tri tuong doi
            if center_x < width / 3:
                position = "bên trái"
            elif center_x > 2 * width / 3:
                position = "bên phải"
            else:
                position = "ở phía trước"

            message = f"Phát hiện {object_name} {position}"

            # Hien thi ket qua
            print(
                f"{message} | Confidence: {confidence:.2f}"
            )

            # Han che doc lap lai
            current_time = time.time()

            if (
                message != last_message
                or current_time - last_speech_time >= 5
            ):
                speech_queue.put(message)
                last_message = message
                last_speech_time = current_time

        # Hien thi khung nhan dien
        display_frame = results[0].plot()

        cv2.imshow("He thong ho tro nguoi khiem thi", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    speech_thread.join()
