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
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ==============================================================================
# 1. CẤU HÌNH & KHỞI TẠO MÔ HÌNH YOLO
# ==============================================================================
import urllib.request

MODEL_PATH = "best.pt"
MODEL_RELEASE_URL = "https://github.com/Nguyen-Thi-Thu1307/BlindAssist/releases/download/v1.0.0/best.pt"

# Tự động tải file best.pt từ GitHub Release nếu chưa có local
if not os.path.exists(MODEL_PATH):
    print(f"[*] Không tìm thấy {MODEL_PATH} cục bộ. Đang tải từ GitHub Release...")
    try:
        urllib.request.urlretrieve(MODEL_RELEASE_URL, MODEL_PATH)
        print(f"[✓] Tải thành công {MODEL_PATH} từ GitHub Release!")
    except Exception as e:
        print(f"[!] Không thể tải {MODEL_PATH} từ Release: {e}")
        # Fallback sang yolo26n.pt nếu tải thất bại
        if os.path.exists("yolo26n.pt"):
            MODEL_PATH = "yolo26n.pt"
        else:
            MODEL_PATH = "yolo26n.pt" # Ultralytics sẽ tự động tải file chuẩn về

print(f"[*] Đang tải mô hình vào YOLO từ: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print(f"[✓] Đã tải mô hình thành công! Danh sách các lớp: {model.names}")

# Danh mục 9 lớp đối tượng Tiếng Việt chuẩn (cho UI & Giọng nói)
VIETNAMESE_NAMES = {
    "projector": "Máy chiếu",
    "desktop_computer": "Máy tính để bàn",
    "laptop": "Laptop",
    "cell_phone": "Điện thoại",
    "whiteboard": "Bảng trắng",
    "remote": "Điều khiển",
    "book": "Sách vở",
    "backpack": "Ba lô",
    "table": "Bàn"
}

# Tên vật thể không dấu dùng khi cần fallback cho OpenCV font
ASCII_NAMES = {
    "projector": "May chieu",
    "desktop_computer": "May tinh de ban",
    "laptop": "Laptop",
    "cell_phone": "Dien thoai",
    "whiteboard": "Bang trang",
    "remote": "Dieu khien",
    "book": "Sach vo",
    "backpack": "Ba lo",
    "table": "Ban"
}

# Bảng màu sắc hiển thị riêng biệt theo từng class (BGR format)
CLASS_COLORS = {
    "projector": (255, 128, 0),        # Xanh lam đậm
    "desktop_computer": (0, 200, 255),  # Vàng nghệ
    "laptop": (76, 175, 80),            # Xanh lục
    "cell_phone": (233, 30, 99),        # Hồng tím
    "whiteboard": (180, 105, 255),      # Tím nhạt
    "remote": (0, 165, 255),            # Cam tươi
    "book": (205, 90, 106),             # Xanh slate
    "backpack": (147, 20, 255),         # Tím đậm
    "table": (42, 42, 165)              # Nâu đất
}

# ==============================================================================
# 2. HỖ TRỢ VẼ TEXT TIẾNG VIỆT CÓ DẤU BẰNG PIL
# ==============================================================================
def get_truetype_font(size=18):
    """Tìm font hệ thống có hỗ trợ Unicode Tiếng Việt"""
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None

UNICODE_FONT = get_truetype_font(size=17)

def draw_labeled_boxes(frame, detections):
    """
    Vẽ khung nhận diện và nhãn Tiếng Việt sắc nét lên frame.
    Sử dụng PIL để hiển thị đầy đủ dấu Tiếng Việt, fallback sang OpenCV nếu cần.
    """
    if not detections:
        return frame

    try:
        # Chuyển OpenCV BGR sang PIL Image để vẽ chữ tiếng Việt có dấu
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color_bgr = det["color"]
            # Chuyển BGR -> RGB cho PIL
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            is_target = det.get("is_target", False)

            thickness = 3 if is_target else 2
            # Vẽ hình chữ nhật bounding box
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color_rgb)

            # Nội dung nhãn: [MỤC TIÊU] Tên (Độ tin cậy)% - Hướng
            target_prefix = "[MỤC TIÊU] " if is_target else ""
            label_text = f"{target_prefix}{det['vietnamese_name']} {int(det['confidence'])}% - {det['position']}"

            # Tính toán kích thước hộp nhãn text
            if UNICODE_FONT:
                bbox_text = draw.textbbox((0, 0), label_text, font=UNICODE_FONT)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            else:
                tw, th = len(label_text) * 9, 16

            box_top = max(0, y1 - th - 10)
            box_bottom = y1
            draw.rectangle([x1, box_top, x1 + tw + 12, box_bottom], fill=color_rgb)
            draw.text((x1 + 6, box_top + 3), label_text, font=UNICODE_FONT, fill=(255, 255, 255))

        # Chuyển lại sang BGR OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        # Fallback an toàn sang cv2.putText (sử dụng ASCII không dấu)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            ascii_label = f"{det['ascii_name']} {int(det['confidence'])}% - {det['ascii_pos']}"
            cv2.putText(frame, ascii_label, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame


# ==============================================================================
# 3. CAMERA CONTROLLER - QUẢN LÝ LUỒNG CAMERA & NHẬN DIỆN THỜI GIAN THỰC
# ==============================================================================
class CameraController:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.confidence_threshold = 0.40
        self.target_class = None  # None: Nhận diện tất cả, hoặc tên class cụ thể
        self.speech_interval = 4.0  # Khoảng thời gian giãn cách đọc thông báo (giây)
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

            # Thử mở camera với CAP_DSHOW trước (tối ưu mở nhanh trên Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback sang mở mặc định
                self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                self.is_running = False
                print("[!] Không thể kết nối với Webcam!")
                return False

            # Đặt độ phân giải ổn định
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_running = True
            self.frame_count = 0
            self.start_fps_time = time.time()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[✓] Camera đã khởi động thành công.")
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_detections = []
            self.current_frame = None
            print("[✓] Camera đã dừng.")

    def set_settings(self, confidence=None, target_class=None, speech_interval=None):
        with self.lock:
            if confidence is not None:
                self.confidence_threshold = max(0.1, min(0.95, float(confidence)))
            if target_class is not None:
                self.target_class = None if target_class in ("all", "", None) else str(target_class)
            if speech_interval is not None:
                self.speech_interval = max(1.0, float(speech_interval))

    def _capture_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.04)
                continue

            # Đo đạc FPS
            now = time.time()
            self.frame_count += 1
            if now - self.start_fps_time >= 1.0:
                self.fps = round(self.frame_count / (now - self.start_fps_time), 1)
                self.frame_count = 0
                self.start_fps_time = now

            height, width = frame.shape[:2]

            # Dự đoán vật thể bằng mô hình YOLO
            results = model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=416,
                device="cpu",
                verbose=False
            )

            detections_data = []
            boxes = results[0].boxes if len(results) > 0 and results[0].boxes is not None else []

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                vi_name = VIETNAMESE_NAMES.get(cls_name, cls_name)
                ascii_name = ASCII_NAMES.get(cls_name, cls_name)

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                center_x = (x1 + x2) / 2

                # Xác định phân vùng không gian 3 hướng
                if center_x < width / 3:
                    position = "bên trái"
                    ascii_pos = "Bên trái"
                    pos_code = "left"
                elif center_x > 2 * width / 3:
                    position = "bên phải"
                    ascii_pos = "Bên phải"
                    pos_code = "right"
                else:
                    position = "ở phía trước"
                    ascii_pos = "Phía trước"
                    pos_code = "center"

                is_target = (self.target_class is not None and cls_name == self.target_class)

                if is_target:
                    color = (0, 255, 0) # Màu xanh neon nổi bật cho mục tiêu đang tìm
                else:
                    color = CLASS_COLORS.get(cls_name, (0, 140, 255))

                item = {
                    "class_name": cls_name,
                    "vietnamese_name": vi_name,
                    "ascii_name": ascii_name,
                    "confidence": round(conf * 100, 1),
                    "position": position,
                    "ascii_pos": ascii_pos,
                    "pos_code": pos_code,
                    "is_target": is_target,
                    "bbox": [x1, y1, x2, y2],
                    "color": color
                }

                # Nếu đang lọc theo mục tiêu thì chỉ lấy đối tượng mục tiêu
                if self.target_class is None or is_target:
                    detections_data.append(item)

            # Vẽ bounding boxes lên khung hình
            annotated_frame = draw_labeled_boxes(frame, detections_data)

            # Xác định thông báo giọng nói ưu tiên
            if detections_data:
                # Nếu có đối tượng mục tiêu thì ưu tiên đọc mục tiêu trước
                if self.target_class:
                    target_items = [d for d in detections_data if d["is_target"]]
                    best_item = max(target_items, key=lambda x: x["confidence"]) if target_items else None
                else:
                    best_item = max(detections_data, key=lambda x: x["confidence"])

                if best_item:
                    announcement = f"Phát hiện {best_item['vietnamese_name']} {best_item['position']}"
                    current_t = time.time()
                    if (announcement != self.last_speech_message) or (current_t - self.last_speech_time >= self.speech_interval):
                        self.latest_announcement = announcement
                        self.last_speech_message = announcement
                        self.last_speech_time = current_t

            with self.lock:
                self.latest_detections = detections_data
                self.current_frame = annotated_frame

            time.sleep(0.01)

    def get_jpeg_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ret:
                return None
            return jpeg.tobytes()

camera = CameraController()

def generate_video_stream():
    """Generator sinh chuỗi multipart MJPEG cho trình duyệt"""
    while True:
        if not camera.is_running:
            # Tạo frame ảnh chờ trang nhã khi camera chưa bật
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder[:] = (20, 24, 33) # Nền tối hiện đại
            cv2.putText(
                placeholder,
                "Camera dang tam dung.",
                (170, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (160, 175, 200),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                placeholder,
                "Nhan 'Bat dau nhan dien' de kich hoat.",
                (120, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 115, 140),
                1,
                cv2.LINE_AA
            )
            _, jpeg = cv2.imencode(".jpg", placeholder)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.4)
            continue

        frame_bytes = camera.get_jpeg_frame()
        if frame_bytes is None:
            time.sleep(0.03)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)


# ==============================================================================
# 4. CÁC ĐƯỜNG DẪN WEB VÀ API ROUTES
# ==============================================================================
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
        "message": "Đã khởi động camera" if success else "Không thể mở camera. Vui lòng kiểm tra quyền truy cập webcam!"
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
        return jsonify({"success": False, "message": "Không tìm thấy file ảnh tải lên"}), 400

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

        boxes = results[0].boxes if len(results) > 0 and results[0].boxes is not None else []
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
                ascii_pos = "Bên trái"
                pos_code = "left"
            elif center_x > 2 * width / 3:
                pos = "bên phải"
                ascii_pos = "Bên phải"
                pos_code = "right"
            else:
                pos = "ở phía trước"
                ascii_pos = "Phía trước"
                pos_code = "center"

            color = CLASS_COLORS.get(cls_name, (0, 200, 255))
            item = {
                "class_name": cls_name,
                "vietnamese_name": vi_name,
                "ascii_name": ascii_name,
                "confidence": round(conf * 100, 1),
                "position": pos,
                "ascii_pos": ascii_pos,
                "pos_code": pos_code,
                "is_target": False,
                "bbox": [x1, y1, x2, y2],
                "color": color
            }
            detections.append(item)

        # Vẽ các hộp kết quả với font tiếng Việt sắc nét
        annotated_img = draw_labeled_boxes(img, detections)

        ret, buffer = cv2.imencode(".jpg", annotated_img)
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

# Bộ nhớ đệm cache giọng đọc đã sinh để phát tức thì
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
        return jsonify({"error": "Thiếu nội dung văn bản"}), 400

    cache_key = f"{voice}:{text}"
    if cache_key in TTS_CACHE:
        return Response(TTS_CACHE[cache_key], mimetype="audio/mpeg")

    async def generate_speech():
        for attempt in range(2):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice)
                byte_io = BytesIO()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        byte_io.write(chunk["data"])
                byte_io.seek(0)
                data = byte_io.read()
                if len(data) > 0:
                    return data
            except Exception:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.2)
        return b""

    try:
        audio_data = asyncio.run(generate_speech())
        if len(audio_data) > 0:
            if len(TTS_CACHE) > 300:
                TTS_CACHE.clear()
            TTS_CACHE[cache_key] = audio_data
            return Response(audio_data, mimetype="audio/mpeg")
        else:
            return jsonify({"error": "Không thể tạo âm thanh"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================================================================
# 5. KHỞI CHẠY SERVER
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  BlindAssist AI Vision Web Server")
    print(f"  Truy cập hệ thống tại: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
