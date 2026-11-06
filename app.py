import os
import cv2
import time
import json
import base64
import asyncio
import threading
from io import BytesIO
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Khởi tạo mô hình YOLO
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy file mô hình {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# Tên vật thể Tiếng Việt chuẩn (cho UI & Giọng nói)
VIETNAMESE_NAMES = {
    "projector": "Máy chiếu",
    "desktop_computer": "Máy tính để bàn",
    "laptop": "Laptop",
    "cell_phone": "Điện thoại"
}

# Tên vật thể không dấu cho OpenCV putText (tránh lỗi font Hershey)
ASCII_NAMES = {
    "projector": "May chieu",
    "desktop_computer": "May tinh de ban",
    "laptop": "Laptop",
    "cell_phone": "Dien thoai"
}

# Màu sắc hiển thị theo class (BGR cho OpenCV)
CLASS_COLORS = {
    "projector": (255, 128, 0),        # Xanh lam đậm
    "desktop_computer": (0, 200, 255),  # Vàng nghệ
    "laptop": (76, 175, 80),            # Xanh lục
    "cell_phone": (233, 30, 99)         # Hồng tím
}

class CameraController:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.confidence_threshold = 0.45
        self.target_class = None  # None: nhận diện tất cả; hoặc tên class cụ thể
        self.speech_interval = 4.0  # Giây giữa các lần thông báo
        self.last_speech_time = 0
        self.last_speech_message = ""
        self.latest_detections = []
        self.latest_announcement = ""
        self.fps = 0.0
        self.frame_count = 0
        self.start_fps_time = time.time()
        self.thread = None

    def start(self):
        with self.lock:
            if self.is_running:
                return True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.is_running = False
                return False
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.latest_detections = []
            self.current_frame = None

    def set_settings(self, confidence=None, target_class=None, speech_interval=None):
        with self.lock:
            if confidence is not None:
                self.confidence_threshold = max(0.1, min(0.95, float(confidence)))
            if target_class is not None:
                self.target_class = None if target_class == "all" or not target_class else str(target_class)
            if speech_interval is not None:
                self.speech_interval = max(1.0, float(speech_interval))

    def _capture_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Tính toán FPS
            now = time.time()
            self.frame_count += 1
            if now - self.start_fps_time >= 1.0:
                self.fps = round(self.frame_count / (now - self.start_fps_time), 1)
                self.frame_count = 0
                self.start_fps_time = now

            height, width = frame.shape[:2]

            # Dự đoán với YOLO
            results = model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False
            )

            detections_data = []
            boxes = results[0].boxes if len(results) > 0 else []

            # Vẽ và trích xuất thông tin từng vật thể
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                vi_name = VIETNAMESE_NAMES.get(cls_name, cls_name)
                ascii_name = ASCII_NAMES.get(cls_name, cls_name)

                # Tọa độ bounding box
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                center_x = (x1 + x2) / 2

                # Xác định phương hướng
                if center_x < width / 3:
                    position = "bên trái"
                    ascii_pos = "Trai"
                    pos_code = "left"
                elif center_x > 2 * width / 3:
                    position = "bên phải"
                    ascii_pos = "Phai"
                    pos_code = "right"
                else:
                    position = "ở phía trước"
                    ascii_pos = "Phia truoc"
                    pos_code = "center"

                is_target = (self.target_class is not None and cls_name == self.target_class)

                detection_item = {
                    "class_name": cls_name,
                    "vietnamese_name": vi_name,
                    "confidence": round(conf * 100, 1),
                    "position": position,
                    "pos_code": pos_code,
                    "is_target": is_target,
                    "bbox": [x1, y1, x2, y2]
                }

                # Nếu đang lọc theo mục tiêu thì ưu tiên chỉ lưu/vẽ mục tiêu
                if self.target_class is None or is_target:
                    detections_data.append(detection_item)

                # Màu sắc vẽ khung
                if is_target:
                    color = (0, 255, 0) # Màu xanh neon nổi bật cho mục tiêu
                    thickness = 3
                else:
                    color = CLASS_COLORS.get(cls_name, (0, 140, 255))
                    thickness = 2

                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Nhãn hiển thị trên OpenCV (dùng ASCII name để hiển thị rõ nét không lỗi font)
                label_text = f"{ascii_name} {int(conf * 100)}% - {ascii_pos}"
                if is_target:
                    label_text = f"[MUC TIEU] " + label_text

                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                y_text_top = max(0, y1 - th - 8)
                cv2.rectangle(frame, (x1, y_text_top), (x1 + tw + 10, y1), color, -1)
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Cập nhật thông tin thông báo giọng nói
            if detections_data:
                # Chọn vật thể ưu tiên: mục tiêu nếu có, hoặc vật thể có độ tin cậy cao nhất
                if self.target_class:
                    target_boxes = [d for d in detections_data if d["is_target"]]
                    best_item = max(target_boxes, key=lambda x: x["confidence"]) if target_boxes else None
                else:
                    best_item = max(detections_data, key=lambda x: x["confidence"])

                if best_item:
                    msg = f"Phát hiện {best_item['vietnamese_name']} {best_item['position']}"
                    current_t = time.time()
                    if msg != self.last_speech_message or (current_t - self.last_speech_time >= self.speech_interval):
                        self.latest_announcement = msg
                        self.last_speech_message = msg
                        self.last_speech_time = current_t

            with self.lock:
                self.latest_detections = detections_data
                self.current_frame = frame.copy()

            time.sleep(0.01)

    def get_jpeg_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                return None
            return jpeg.tobytes()

camera = CameraController()

def generate_video_stream():
    """Generator sinh frame MJPEG cho trình duyệt"""
    while True:
        if not camera.is_running:
            # Tạo frame ảnh chờ màu đen khi camera chưa bật
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Camera dang tat. Nhan Bat dau de xem luong.",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )
            _, jpeg = cv2.imencode(".jpg", placeholder)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        frame_bytes = camera.get_jpeg_frame()
        if frame_bytes is None:
            time.sleep(0.03)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template("index.html", supported_classes=VIETNAMESE_NAMES)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    success = camera.start()
    return jsonify({
        "success": success,
        "is_running": camera.is_running,
        "message": "Đã khởi động camera" if success else "Không thể mở camera. Vui lòng kiểm tra webcam!"
    })

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    camera.stop()
    return jsonify({
        "success": True,
        "is_running": False,
        "message": "Đã dừng camera"
    })

@app.route("/api/status", methods=["GET"])
def get_status():
    with camera.lock:
        return jsonify({
            "is_running": camera.is_running,
            "fps": camera.fps,
            "confidence_threshold": camera.confidence_threshold,
            "target_class": camera.target_class,
            "target_name": VIETNAMESE_NAMES.get(camera.target_class, "Tất cả") if camera.target_class else "Tất cả",
            "detections": camera.latest_detections,
            "latest_announcement": camera.latest_announcement,
            "speech_interval": camera.speech_interval
        })

@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json() or {}
    confidence = data.get("confidence")
    target_class = data.get("target_class")
    speech_interval = data.get("speech_interval")

    camera.set_settings(
        confidence=confidence,
        target_class=target_class,
        speech_interval=speech_interval
    )

    return jsonify({
        "success": True,
        "confidence": camera.confidence_threshold,
        "target_class": camera.target_class,
        "target_name": VIETNAMESE_NAMES.get(camera.target_class, "Tất cả") if camera.target_class else "Tất cả",
        "speech_interval": camera.speech_interval
    })

@app.route("/api/detect_image", methods=["POST"])
def detect_uploaded_image():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Không tìm thấy file tải lên"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Chưa chọn file"}), 400

    try:
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "message": "File ảnh không hợp lệ"}), 400

        height, width = img.shape[:2]
        results = model.predict(source=img, conf=camera.confidence_threshold, imgsz=640, verbose=False)

        boxes = results[0].boxes if len(results) > 0 else []
        detections = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names.get(cls_id, f"Class {cls_id}")
            vi_name = VIETNAMESE_NAMES.get(cls_name, cls_name)
            ascii_name = ASCII_NAMES.get(cls_name, cls_name)

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            center_x = (x1 + x2) / 2

            if center_x < width / 3:
                pos = "bên trái"
                ascii_pos = "Trai"
            elif center_x > 2 * width / 3:
                pos = "bên phải"
                ascii_pos = "Phai"
            else:
                pos = "ở phía trước"
                ascii_pos = "Phia truoc"

            detections.append({
                "class_name": cls_name,
                "vietnamese_name": vi_name,
                "confidence": round(conf * 100, 1),
                "position": pos
            })

            # Vẽ lên ảnh
            color = CLASS_COLORS.get(cls_name, (0, 200, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"{ascii_name} {int(conf * 100)}% - {ascii_pos}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "total_detected": len(detections),
            "detections": detections,
            "image_data": f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/classes", methods=["GET"])
def get_supported_classes():
    return jsonify({
        "classes": [
            {"id": k, "name": v, "code": k} for k, v in VIETNAMESE_NAMES.items()
        ]
    })

# Cache âm thanh đã tạo để phát tức thì không trễ mạng
TTS_CACHE = {}

@app.route("/api/tts", methods=["GET", "POST"])
def tts_audio():
    """Tạo audio giọng đọc tiếng Việt qua edge-tts với bộ nhớ đệm (cache)"""
    import edge_tts
    if request.method == "POST" and request.is_json:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        voice = data.get("voice", "vi-VN-HoaiMyNeural")
    else:
        text = request.args.get("text", "").strip()
        voice = request.args.get("voice", "vi-VN-HoaiMyNeural")

    if not text:
        return jsonify({"error": "Thiếu nội dung text"}), 400

    # Kiểm tra trong cache trước
    cache_key = f"{voice}:{text}"
    if cache_key in TTS_CACHE:
        return Response(TTS_CACHE[cache_key], mimetype="audio/mpeg")

    async def generate_speech():
        for attempt in range(2):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice)
                byteIO = BytesIO()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        byteIO.write(chunk["data"])
                byteIO.seek(0)
                data = byteIO.read()
                if len(data) > 0:
                    return data
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.2)
        return b""

    try:
        audio_data = asyncio.run(generate_speech())
        # Lưu vào cache (tối đa 200 câu thông dụng)
        if len(TTS_CACHE) > 200:
            TTS_CACHE.clear()
        TTS_CACHE[cache_key] = audio_data
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
