# 👁️ BlindAssist - Hệ Thống Trợ Thị Thông Minh Hỗ Trợ Người Khiếm Thị Bằng AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?logo=yolo&logoColor=black" alt="YOLO" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/TTS-Edge--TTS%20Vietnamese-green" alt="Edge TTS" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?logo=opencv&logoColor=white" alt="OpenCV" />
</p>

**BlindAssist** là hệ thống trợ thị thông minh ứng dụng mô hình thị giác máy tính **YOLO** kết hợp công nghệ **Text-to-Speech (TTS)** tiếng Việt thời gian thực. Hệ thống giúp người khiếm thị nhận biết các vật thể xung quanh (máy tính, laptop, điện thoại, sách, bàn, bảng viết,...), xác định phương hướng không gian (trái, phải, phía trước) và nhận thông báo giọng nói tự nhiên, tức thì.

---

## 🌟 Tính Năng Nổi Bật

- 🎯 **Nhận diện 9 lớp vật thể thiết yếu**: Được huấn luyện tối ưu cho các vật thể quen thuộc trong môi trường học tập và làm việc.
- 🧭 **Định vị phương hướng không gian**: Tự động phân tích tọa độ của vật thể trên khung hình để đưa ra vị trí tương đối: *Bên trái*, *Ở phía trước*, hoặc *Bên phải*.
- 🔊 **Thông báo giọng nói tiếng Việt tự nhiên**: Tích hợp công nghệ tổng hợp giọng đọc tiếng Việt (`Edge-TTS` với giọng đọc tự nhiên `vi-VN-HoaiMyNeural` & Web Speech), hỗ trợ cache âm thanh giúp phản hồi gần như không độ trễ.
- ⏱️ **Cơ chế chống đọc lặp thông minh (Cooldown & Queue)**: Tránh phát âm thanh dồn dập, chỉ phát thông báo khi có vật thể mới hoặc sau một khoảng thời gian thiết lập.
- 🌐 **Giao diện Web Dashboard trực quan (Flask & Modern UI)**:
  - Xem luồng video camera trực tiếp (Live MJPEG Stream).
  - Tải ảnh tĩnh từ máy tính lên để phân tích (`/api/detect_image`).
  - Tùy chỉnh trực tiếp ngưỡng tin cậy (Confidence Threshold) và tần suất đọc âm thanh (Speech Interval).
  - Lọc nhận diện theo mục tiêu cụ thể hoặc tất cả các lớp.
  - Bảng ghi nhận thông báo thời gian thực (Live Event Logs).

---

## 🏷️ Danh Sách 9 Lớp Đối Tượng Nhận Diện

| STT | Class ID | Tên Tiếng Anh | Tên Tiếng Việt | Mô Tả |
|:---:|:---:|:---|:---|:---|
| 1 | 0 | `projector` | **Máy chiếu** | Thiết bị máy chiếu phòng họp / lớp học |
| 2 | 1 | `desktop_computer` | **Máy tính để bàn** | Màn hình, case PC để bàn |
| 3 | 2 | `laptop` | **Laptop** | Máy tính xách tay |
| 4 | 3 | `cell_phone` | **Điện thoại** | Điện thoại di động, smartphone |
| 5 | 4 | `whiteboard` | **Bảng trắng** | Bảng viết phấn / bút lông |
| 6 | 5 | `remote` | **Điều khiển** | Điều khiển từ xa (máy chiếu, điều hòa, TV) |
| 7 | 6 | `book` | **Sách vở** | Sách giáo khoa, tài liệu đọc |
| 8 | 7 | `backpack` | **Balo / Cặp** | Balo, túi xách |
| 9 | 8 | `table` | **Bàn** | Bàn học, bàn làm việc |

---

## 📁 Cấu Trúc Thư Mục Dự Án

```plaintext
YOLO_Project/
├── combined_dataset_9classes/     # Dữ liệu huấn luyện (train, valid, test, data.yaml)
│   ├── train/images & labels      # Tập huấn luyện
│   ├── valid/images & labels      # Tập kiểm định
│   ├── test/images & labels       # Tập kiểm thử
│   └── data.yaml                  # File cấu hình dữ liệu cho YOLO
├── static/                        # Tài nguyên Web UI
│   ├── css/
│   │   └── style.css              # Giao diện hiện đại, hỗ trợ Dark/Light theme
│   ├── js/
│   │   └── script.js              # Xử lý sự kiện camera, gọi API, phát âm thanh TTS
│   └── audio/                     # Thư mục chứa hiệu ứng âm thanh (nếu có)
├── templates/
│   └── index.html                 # Giao diện Web BlindAssist Dashboard
├── .venv/                         # Môi trường ảo Python
├── app.py                         # Ứng dụng Web Flask chính & Bộ xử lý Camera
├── best.pt                        # Trọng số mô hình đã huấn luyện chuẩn 9 classes
├── yolo26n.pt                     # Mô hình cơ sở (Pretrained YOLO model)
├── train.py                       # Script huấn luyện/tinh chỉnh mô hình YOLO
├── test_model.py                  # Script kiểm tra file weights và danh sách class
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu Cầu Tiên Quyết
- Hệ điều hành: Windows, macOS hoặc Linux.
- **Python**: Phiên bản 3.10 hoặc 3.11.
- Webcam (hoặc camera ngoài kết nối qua USB).
- Kết nối Internet (phục vụ tổng hợp giọng nói online qua Edge-TTS).

### 2. Cài Đặt Môi Trường

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment):**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Cài đặt các gói thư viện phụ thuộc:**
   ```bash
   pip install --upgrade pip
   ```

### 3. Kiểm Tra Mô Hình
Trước khi khởi động ứng dụng web, chạy script sau để xác nhận mô hình `best.pt` hoạt động bình thường:
```bash
python test_model.py
```
*Kết quả hiển thị danh sách 9 classes và thông báo "Mô hình đã tải thành công!" là hợp lệ.*

---

## 💻 Khởi Chạy Ứng Dụng

### Cách 1: Chạy Web Application (Khuyên dùng)
Khởi chạy máy chủ Flask để mở giao diện bảng điều khiển:
```bash
python app.py
```
Sau đó mở trình duyệt web và truy cập địa chỉ:
```
http://localhost:5000
```

**Các thao tác trên Giao Diện Web:**
- Bấm **"Bật Camera"**: Bắt đầu nhận diện qua webcam.
- Bấm **"Tắt Camera"**: Tạm dừng luồng quay để tiết kiệm tài nguyên.
- Mục **"Mục tiêu nhận diện"**: Chọn lọc đối tượng cần ưu tiên thông báo (ví dụ chỉ tìm "Điện thoại" hoặc "Laptop").
- Mục **"Ngưỡng tin cậy"**: Kéo thanh trượt điều chỉnh độ nhạy (khuyến nghị `0.45` - `0.60`).
- Mục **"Khoảng cách giọng nói"**: Đặt thời gian nghỉ giữa các lần đọc thông báo (mặc định `4.0` giây).
- Khu vực **"Tải ảnh lên"**: Kéo thả ảnh bất kỳ để nhận diện ngay lập tức mà không cần bật webcam.

---

### 4. Hướng dẫn kết nối với điện thoại (Android 11 trở lên)
## Kết nối điện thoại Android với máy tính qua ADB Wireless Debugging

### 4.1. Giới thiệu

Dự án sử dụng ADB (Android Debug Bridge) để kết nối điện thoại Android với máy tính thông qua mạng Wi-Fi. Việc kết nối không dây cho phép cài đặt, chạy thử và kiểm tra ứng dụng Android trực tiếp trên điện thoại mà không cần sử dụng cáp USB.

Trong dự án này, môi trường kết nối gồm:

- **Máy tính:** Windows 10/11.
- **IDE:** Visual Studio Code.
- **Điện thoại:** Samsung Galaxy A13, Android 14.
- **Công cụ:** Android SDK Platform Tools (ADB).
- **Phương thức kết nối:** Wireless Debugging (ADB qua Wi-Fi).

### 4.2. Điều kiện trước khi kết nối

Trước khi thực hiện, cần đảm bảo:

1. Máy tính và điện thoại kết nối cùng một mạng Wi-Fi.
2. Điện thoại sử dụng Android 11 trở lên và hỗ trợ Wireless Debugging.
3. Đã bật Developer Options (Tùy chọn nhà phát triển) trên điện thoại.
4. Đã tải và giải nén Android SDK Platform Tools trên máy tính.
5. Không sử dụng mạng Wi-Fi khách (Guest Wi-Fi) có tính năng cô lập các thiết bị.

### 4.3. Cài đặt ADB trên Windows

**Bước 1: Tải Android SDK Platform Tools**

Truy cập trang chính thức của Android:

https://developer.android.com/tools/releases/platform-tools

Tải phiên bản dành cho Windows và giải nén vào thư mục:

```text
D:\Android\platform-tools
```

Sau khi giải nén, kiểm tra thư mục có tệp:

```text
adb.exe
```

**Bước 2: Kiểm tra ADB**

Mở PowerShell và nhập:

```powershell
& "D:\Android\platform-tools\adb.exe" version
```

Nếu cài đặt thành công, màn hình sẽ hiển thị thông tin phiên bản Android Debug Bridge.

**Bước 3: Thiết lập PATH trong Git Bash (VS Code)**

Nếu sử dụng Git Bash làm Terminal trong VS Code, nhập:

```bash
export PATH="/d/Android/platform-tools:$PATH"
```

Kiểm tra:

```bash
adb version
```

Nếu hiển thị phiên bản ADB thì có thể sử dụng các lệnh ADB trực tiếp trong Git Bash.

*Lưu ý: Lệnh export trên chỉ có hiệu lực trong phiên Terminal hiện tại. Khi mở Terminal mới, cần chạy lại nếu chưa thiết lập PATH cố định.*

### 4.4. Bật chế độ Wireless Debugging trên điện thoại

Thực hiện trên điện thoại:
**Bước 1: Bật Developer Options**

Nếu chưa bật Tùy chọn nhà phát triển:

1. Mở Cài đặt.
2. Chọn Thông tin điện thoại.
3. Chọn Thông tin phần mềm.
4. Nhấn liên tục 7 lần vào Số hiệu bản tạo (Build number).
5. Nhập mã PIN nếu được yêu cầu.

**Bước 2: Bật chế độ gỡ lỗi**

1. Mở Cài đặt.
2. Chọn Tùy chọn nhà phát triển.
3. Bật Gỡ lỗi qua USB (USB debugging).
4. Bật Gỡ lỗi qua Wi-Fi (Wireless debugging).

**Bước 3: Mở chức năng ghép nối**

1. Truy cập Gỡ lỗi qua Wi-Fi.
2. Chọn Ghép nối thiết bị bằng mã ghép nối (Pair device with pairing code).
3. Điện thoại sẽ hiển thị:
   - Địa chỉ IP.
   - Cổng ghép nối (Pairing port).
   - Mã ghép nối gồm 6 chữ số.

Giữ màn hình này mở trong quá trình ghép nối.

### 4.5. Ghép nối điện thoại với máy tính

Có thể thực hiện bằng PowerShell hoặc Git Bash.

**Bước 1: Khởi động ADB Server**

Mở PowerShell và nhập:

```powershell
& "D:\Android\platform-tools\adb.exe" kill-server
& "D:\Android\platform-tools\adb.exe" start-server
```

**Bước 2: Thực hiện ghép nối**

Sử dụng địa chỉ IP và cổng ghép nối đang hiển thị trên điện thoại:

```powershell
& "D:\Android\platform-tools\adb.exe" pair IP:PAIRING_PORT
```

Ví dụ:

```powershell
& "D:\Android\platform-tools\adb.exe" pair 192.168.1.111:39151
```

**Bước 3: Nhập mã ghép nối**

Khi PowerShell hiển thị:

```text
Enter pairing code:
```

Nhập mã 6 chữ số đang hiển thị trên điện thoại rồi nhấn Enter.

Nếu thành công, kết quả sẽ tương tự:

```text
Successfully paired to 192.168.1.111:39151
```

*Lưu ý: IP và cổng trong ví dụ chỉ mang tính minh họa. Cần sử dụng thông tin mới nhất trên điện thoại. Mã ghép nối không được chia sẻ công khai.*

### 4.6. Kết nối điện thoại sau khi ghép nối

Sau khi ghép nối thành công, quay lại màn hình Gỡ lỗi qua Wi-Fi chính trên điện thoại.

Tìm địa chỉ IP và cổng kết nối (IP address & Port). Cổng này có thể khác cổng ghép nối ở bước trước.

Thực hiện lệnh:

```bash
adb connect IP:CONNECT_PORT
```

Ví dụ:

```bash
adb connect 192.168.1.111:42001
```

Nếu kết nối thành công, kết quả sẽ tương tự:

```text
connected to 192.168.1.111:42001
```

### 4.7. Kiểm tra thiết bị đã kết nối

Nhập lệnh:

```bash
adb devices
```

Nếu kết nối thành công, danh sách thiết bị sẽ hiển thị tương tự:

```text
List of devices attached
192.168.1.111:42001    device
```

Trong đó:

- `device`: Điện thoại đã kết nối và sẵn sàng sử dụng.
- `offline`: Thiết bị đã được nhận diện nhưng chưa sẵn sàng giao tiếp.
- Không có thiết bị: Chưa kết nối thành công.

Khi thiết bị hiển thị trạng thái `device`, có thể sử dụng ADB để cài đặt, chạy và kiểm tra ứng dụng Android.

### 4.8. Một số lệnh ADB thường sử dụng

| Lệnh | Chức năng |
|---|---|
| `adb version` | Kiểm tra phiên bản ADB |
| `adb devices` | Liệt kê thiết bị đã kết nối |
| `adb pair IP:PORT` | Ghép nối điện thoại |
| `adb connect IP:PORT` | Kết nối điện thoại |
| `adb disconnect` | Ngắt kết nối không dây |
| `adb kill-server` | Dừng ADB Server |
| `adb start-server` | Khởi động ADB Server |
| `adb install app.apk` | Cài đặt ứng dụng APK |
| `adb logcat` | Theo dõi nhật ký hoạt động của ứng dụng |

### 4.9. Xử lý một số lỗi thường gặp

**Lỗi 1: `adb: command not found`**

Nguyên nhân: Git Bash chưa nhận diện được đường dẫn chứa adb.exe.

Cách khắc phục:

```bash
export PATH="/d/Android/platform-tools:$PATH"
```

**Lỗi 2: `protocol fault (couldn't read status message): No error`**

Nguyên nhân có thể liên quan đến quá trình ghép nối ADB, mạng Wi-Fi hoặc môi trường kết nối.

Cách khắc phục:

1. Kiểm tra máy tính và điện thoại có cùng Wi-Fi.
2. Đảm bảo đã bật cả USB debugging và Wireless debugging.
3. Tắt VPN nếu đang sử dụng.
4. Khởi động lại ADB Server.
5. Tạo mã ghép nối mới trên điện thoại.
6. Thử ghép nối bằng PowerShell với đường dẫn adb.exe đầy đủ.

**Lỗi 3: `TcpTestSucceeded: False`**

Nguyên nhân: Máy tính chưa truy cập được cổng ghép nối của điện thoại.

Cách kiểm tra trên PowerShell:

```powershell
Test-NetConnection IP -Port PAIRING_PORT
```

Nếu kết quả là `False`, cần kiểm tra:

- Hai thiết bị có thực sự cùng mạng Wi-Fi hay không.
- Router có bật AP Isolation (cô lập thiết bị) hay không.
- Tường lửa Windows có chặn kết nối ADB hay không.
- Địa chỉ IP và cổng có còn chính xác hay không.

**Lỗi 4: `failed to connect`**

Nguyên nhân có thể do sử dụng sai cổng kết nối hoặc điện thoại đã thay đổi địa chỉ IP.

Cách khắc phục:

1. Mở lại Wireless Debugging trên điện thoại.
2. Kiểm tra IP và cổng kết nối hiện tại.
3. Thực hiện lại lệnh `adb connect`.

### 4.10. Sử dụng ADB để chạy ứng dụng Android

Sau khi điện thoại kết nối thành công, có thể sử dụng ADB để cài đặt ứng dụng:

```bash
adb install app-debug.apk
```

Theo dõi nhật ký hoạt động:

```bash
adb logcat
```

Kiểm tra thiết bị:

```bash
adb devices
```

Khi phát triển ứng dụng, có thể kết hợp VS Code, Gradle và ADB để biên dịch APK, cài đặt và kiểm thử trực tiếp trên điện thoại Samsung Galaxy A13.

### 4.11. Lưu ý

- Địa chỉ IP và cổng kết nối có thể thay đổi khi khởi động lại Wi-Fi hoặc Wireless Debugging.
- Cổng ghép nối và cổng kết nối là hai cổng khác nhau.
- Chỉ cần ghép nối lại khi thiết bị yêu cầu hoặc thông tin ghép nối không còn hiệu lực.
- Sau khi đã ghép nối, cần sử dụng cổng kết nối trên màn hình Wireless Debugging chính để chạy lệnh `adb connect`.
- Không chia sẻ mã ghép nối hoặc bật gỡ lỗi không dây trên mạng Wi-Fi công cộng không đáng tin cậy.

**Kết quả mong đợi:** Máy tính có thể nhận diện và kết nối không dây với Android thông qua ADB, tạo điều kiện thuận lợi cho quá trình phát triển và kiểm thử ứng dụng Android bằng VS Code.

## 🏋️ Huấn Luyện Mô Hình (Training)

Nếu bạn muốn huấn luyện lại hoặc train thêm trên tập dữ liệu mới:

1. Đặt ảnh và nhãn theo chuẩn YOLO vào thư mục `combined_dataset_9classes/`.
2. Kiểm tra đường dẫn và số lớp trong file `combined_dataset_9classes/data.yaml`.
3. Chạy lệnh:
   ```bash
   python train.py
   ```
4. Mô hình sau khi train xong sẽ được lưu tại thư mục `runs/detect/yolo26n_9classes/weights/best.pt`. Copy file `best.pt` này ra thư mục gốc dự án để sử dụng.

---

## 📦 Danh Sách Thư Viện (requirements.txt)

```plaintext
flask>=3.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
edge-tts>=6.1.0
pillow>=10.0.0
numpy>=1.24.0
gunicorn>=21.0.0
```

---

## 🛠️ Luồng Xử Lý Của Hệ Thống (Workflow)

```
[ Camera / Ảnh Tải Lên ]
          │
          ▼
[ Tiền xử lý khung hình (OpenCV) ]
          │
          ▼
[ Dự đoán đối tượng (YOLO: best.pt) ]
   ├── Bounding Box (Tọa độ)
   ├── Class ID & Tên đối tượng
   └── Độ tin cậy (Confidence)
          │
          ▼
[ Xác định vị trí không gian ]
   ├── Trái: X_center < Width / 3
   ├── Trước: Width / 3 <= X_center <= 2*Width / 3
   └── Phải: X_center > 2*Width / 3
          │
          ▼
[ Tạo câu thông báo Tiếng Việt ]
   Ví dụ: "Phát hiện Laptop ở phía trước"
          │
          ▼
[ Bộ lọc chống lặp & Bộ nhớ đệm (Cache) ]
          │
          ▼
[ Chuyển văn bản thành giọng nói (Edge-TTS) ]
          │
          ▼
[ Phát âm thanh ra Loa / Tai nghe cho người dùng ]
```

---

## 🤝 Đóng Góp & Phát Triển
Mọi đóng góp nhằm cải thiện mô hình, tối ưu hóa tốc độ nhận diện hoặc mở rộng các tính năng trợ năng cho người khiếm thị đều được hoan nghênh.
1. Fork dự án
2. Tạo nhánh tính năng mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở một Pull Request

---

## 📄 Bản Quyền
Dự án được xây dựng với mục tiêu phi thương mại, phục vụ nghiên cứu học thuật và hỗ trợ cộng đồng.
