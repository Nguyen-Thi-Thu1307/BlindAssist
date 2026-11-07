import cv2
import time
import os
import asyncio
import queue
import threading
import tempfile
import pygame
import edge_tts

from ultralytics import YOLO

# 1. CẤU HÌNH

MODEL_PATH = "best.pt"
VOICE = "vi-VN-HoaiMyNeural"
CONFIDENCE = 0.4
IMAGE_SIZE = 416
COOLDOWN = 5  # Giây giữa hai lần đọc cùng một vật thể

# Tên class tiếng Việt
CLASS_VI = {
    "projector": "máy chiếu",
    "desktop_computer": "máy tính để bàn",
    "laptop": "máy tính xách tay",
    "cell_phone": "điện thoại",
    "whiteboard": "bảng trắng",
    "remote": "điều khiển từ xa",
    "book": "quyển sách",
    "backpack": "ba lô",
    "table": "cái bàn"
}

# Hàng đợi giọng nói
speech_queue = queue.Queue(maxsize=1)

# 2. KHỞI TẠO ÂM THANH
pygame.mixer.init()

async def create_speech(text, file_path):
    communicate = edge_tts.Communicate(
        text,
        VOICE
    )
    await communicate.save(file_path)


def speech_worker():
    while True:
        text = speech_queue.get()

        if text is None:
            break

        file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            ) as temp:
                file_path = temp.name

            asyncio.run(create_speech(text, file_path))

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()

        except Exception as e:
            print("Lỗi giọng nói:", e)

        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

            speech_queue.task_done()


def speak(text):
    # Không để hàng đợi bị đầy và phát âm thanh liên tục
    if speech_queue.empty():
        speech_queue.put_nowait(text)


# Chạy TTS trên luồng riêng
voice_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)
voice_thread.start()

# 3. TẢI MÔ HÌNH YOLO
print("Đang tải mô hình YOLO26n...")

model = YOLO(MODEL_PATH)

print("Đã tải mô hình thành công!")
print("Các class:", model.names)

# 4. MỞ WEBCAM
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Không thể mở webcam!")
    speech_queue.put(None)
    camera.release()
    pygame.mixer.quit()
    raise SystemExit

print("Webcam đã mở!")
print("Nhấn Q để thoát.")

# Lưu thời điểm đọc gần nhất của từng class
last_spoken = {}

# 5. NHẬN DIỆN VÀ ĐỌC TÊN
try:
    while True:

        ret, frame = camera.read()

        if not ret:
            print("Không lấy được hình ảnh!")
            break

        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # Vẽ khung nhận diện
        annotated_frame = result.plot()

        detected_classes = set()

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detected_classes.add(class_name)

        current_time = time.time()

        for class_name in detected_classes:

            vietnamese_name = CLASS_VI.get(
                class_name,
                class_name
            )

            previous_time = last_spoken.get(
                class_name,
                0
            )

            if current_time - previous_time >= COOLDOWN:

                message = f"Phát hiện {vietnamese_name}"

                print(message)

                speak(message)

                last_spoken[class_name] = current_time

        # Hiển thị webcam
        cv2.imshow(
            "YOLO26n - Nhan dien vat the va giong noi",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    voice_thread.join(timeout=10)

    pygame.mixer.quit()

    print("Đã đóng chương trình!")
