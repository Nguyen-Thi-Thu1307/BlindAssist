/**
 * BlindAssist - AI Vision Web Application
 * Quản lý Camera, Nhận diện YOLO26n, Nhận diện giọng nói & Đọc âm thanh
 */

// ==========================================
// 1. Quản lý trạng thái ứng dụng (State)
// ==========================================
const state = {
    isRunning: false,
    isAudioMuted: false,
    targetClass: 'all',
    targetName: 'Tất cả',
    confidence: 0.45,
    speechInterval: 4.0,
    voice: 'vi-VN-HoaiMyNeural', // Giọng đọc Tiếng Việt chuẩn
    lastSpokenText: '',
    lastSpeechTime: 0,
    statusPollInterval: null,
    isSpeechRecognitionActive: false
};

// Từ điển ánh xạ từ khóa giọng nói tiếng Việt -> Mã class YOLO
const VOICE_KEYWORD_MAP = {
    'điện thoại': 'cell_phone',
    'phone': 'cell_phone',
    'di động': 'cell_phone',
    'laptop': 'laptop',
    'máy tính xách tay': 'laptop',
    'máy tính để bàn': 'desktop_computer',
    'máy tính bàn': 'desktop_computer',
    'máy tính': 'desktop_computer',
    'máy chiếu': 'projector',
    'tất cả': 'all',
    'mọi thứ': 'all'
};

const CLASS_DISPLAY_NAMES = {
    'all': 'Tất cả',
    'cell_phone': 'Điện thoại',
    'laptop': 'Laptop',
    'desktop_computer': 'Máy tính để bàn',
    'projector': 'Máy chiếu'
};

// ==========================================
// 2. DOM Elements
// ==========================================
const DOM = {
    cameraFeed: document.getElementById('cameraFeed'),
    videoContainer: document.getElementById('videoContainer'),
    cameraStandby: document.getElementById('cameraStandby'),
    scanLine: document.getElementById('scanLine'),
    
    // Status badges & HUD
    systemStatusChip: document.getElementById('systemStatusChip'),
    systemStatusText: document.getElementById('systemStatusText'),
    currentTargetName: document.getElementById('currentTargetName'),
    fpsCounter: document.getElementById('fpsCounter'),
    detectedCount: document.getElementById('detectedCount'),
    activeObjectsPill: document.getElementById('activeObjectsPill'),
    
    // Buttons
    btnToggleCamera: document.getElementById('btnToggleCamera'),
    btnToggleText: document.getElementById('btnToggleText'),
    btnVoiceSearch: document.getElementById('btnVoiceSearch'),
    btnVoiceText: document.getElementById('btnVoiceText'),
    btnToggleAudio: document.getElementById('btnToggleAudio'),
    iconAudioOn: document.getElementById('iconAudioOn'),
    iconAudioOff: document.getElementById('iconAudioOff'),
    btnReplayAudio: document.getElementById('btnReplayAudio'),
    btnUploadModal: document.getElementById('btnUploadModal'),
    
    // Voice Feedback UI
    voiceTranscriptBar: document.getElementById('voiceTranscriptBar'),
    voiceTranscriptText: document.getElementById('voiceTranscriptText'),
    voiceWave: document.getElementById('voiceWave'),
    
    // Direction Radar
    directionBadge: document.getElementById('directionBadge'),
    dirLeft: document.getElementById('dirLeft'),
    dirCenter: document.getElementById('dirCenter'),
    dirRight: document.getElementById('dirRight'),
    countLeft: document.getElementById('countLeft'),
    countCenter: document.getElementById('countCenter'),
    countRight: document.getElementById('countRight'),
    
    // Cards & Lists
    announcementText: document.getElementById('announcementText'),
    detectionsList: document.getElementById('detectionsList'),
    targetChips: document.querySelectorAll('.chip'),
    
    // Settings
    confSlider: document.getElementById('confSlider'),
    confValueDisplay: document.getElementById('confValueDisplay'),
    intervalSelect: document.getElementById('intervalSelect'),
    voiceSelect: document.getElementById('voiceSelect'),
    
    // Modal Upload
    uploadModal: document.getElementById('uploadModal'),
    modalBackdrop: document.getElementById('modalBackdrop'),
    btnCloseModal: document.getElementById('btnCloseModal'),
    uploadDropzone: document.getElementById('uploadDropzone'),
    fileInput: document.getElementById('fileInput'),
    uploadResult: document.getElementById('uploadResult'),
    resultImage: document.getElementById('resultImage'),
    resultDetails: document.getElementById('resultDetails')
};

// ==========================================
// 3. Hiệu ứng âm thanh Chime & Tổng hợp giọng nói (TTS)
// ==========================================
class SoundFX {
    constructor() {
        this.audioCtx = null;
    }

    init() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
        }
    }

    // Tiếng bíp thông báo nhẹ (Tone 880Hz -> 1200Hz)
    playTone(frequency = 880, duration = 0.15, type = 'sine') {
        if (state.isAudioMuted) return;
        try {
            this.init();
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(frequency, this.audioCtx.currentTime);
            gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(this.audioCtx.destination);
            osc.start();
            osc.stop(this.audioCtx.currentTime + duration);
        } catch (e) {
            console.warn('Audio tone error:', e);
        }
    }

    // Tiếng chime 2 nốt báo thành công / phát hiện mục tiêu
    playSuccessChime() {
        this.playTone(660, 0.1);
        setTimeout(() => this.playTone(990, 0.2), 110);
    }
}

const sfx = new SoundFX();

// ==========================================
// Bộ máy đọc giọng nói Tiếng Việt chuẩn (Edge-TTS Neural)
// ==========================================
let currentSpeechAudio = null;

function speakMessage(text, force = false) {
    if (state.isAudioMuted || !text) return;

    const now = Date.now();
    if (!force && text === state.lastSpokenText && (now - state.lastSpeechTime) < (state.speechInterval * 1000)) {
        return;
    }

    state.lastSpokenText = text;
    state.lastSpeechTime = now;

    // Dừng audio cũ nếu đang phát dở
    if (currentSpeechAudio) {
        currentSpeechAudio.pause();
        currentSpeechAudio.currentTime = 0;
    }

    // Luôn ưu tiên Edge-TTS Tiếng Việt tự nhiên (Hoài My / Nam Minh)
    const voice = state.voice || 'vi-VN-HoaiMyNeural';
    const audioUrl = `/api/tts?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(voice)}`;

    currentSpeechAudio = new Audio(audioUrl);
    
    currentSpeechAudio.play().catch(err => {
        console.warn('Edge-TTS direct playback issue, checking speech synthesis fallback:', err);
        // Fallback sang SpeechSynthesis CHỈ KHI trình duyệt có cài gói giọng Tiếng Việt
        if ('speechSynthesis' in window) {
            const voices = window.speechSynthesis.getVoices();
            const viVoice = voices.find(v => v.lang.startsWith('vi') || v.lang === 'vi-VN');
            if (viVoice) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.voice = viVoice;
                utterance.lang = 'vi-VN';
                utterance.rate = 1.05;
                window.speechSynthesis.speak(utterance);
            }
        }
    });
}

// ==========================================
// 4. Nhận diện giọng nói (Voice Search)
// ==========================================
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
        state.isSpeechRecognitionActive = true;
        DOM.btnVoiceSearch.classList.add('listening');
        DOM.btnVoiceText.textContent = 'Đang lắng nghe...';
        DOM.voiceWave.classList.add('active');
        DOM.voiceTranscriptText.innerHTML = 'Đang nghe... <em>Hãy nói: "Tìm điện thoại", "Tìm laptop", hoặc "Tất cả"</em>';
        sfx.playTone(520, 0.12);
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.toLowerCase().trim();
        DOM.voiceTranscriptText.innerHTML = `Đã nhận diện: <strong>"${transcript}"</strong>`;
        handleVoiceCommand(transcript);
    };

    recognition.onerror = (event) => {
        console.warn('Speech Recognition error:', event.error);
        DOM.voiceTranscriptText.textContent = 'Không nhận diện được giọng nói. Vui lòng thử lại!';
    };

    recognition.onend = () => {
        state.isSpeechRecognitionActive = false;
        DOM.btnVoiceSearch.classList.remove('listening');
        DOM.btnVoiceText.textContent = 'Tìm kiếm bằng giọng nói';
        DOM.voiceWave.classList.remove('active');
    };
}

function handleVoiceCommand(transcript) {
    // 1. Lệnh điều khiển camera
    if (transcript.includes('bắt đầu') || transcript.includes('bật camera')) {
        if (!state.isRunning) toggleCamera();
        speakMessage('Đã khởi động camera nhận diện', true);
        return;
    }
    if (transcript.includes('dừng') || transcript.includes('tắt camera')) {
        if (state.isRunning) toggleCamera();
        speakMessage('Đã dừng camera', true);
        return;
    }

    // 2. Tìm kiếm vật thể theo từ khóa
    let matchedClass = null;
    for (const [kw, cls] of Object.entries(VOICE_KEYWORD_MAP)) {
        if (transcript.includes(kw)) {
            matchedClass = cls;
            break;
        }
    }

    if (matchedClass) {
        setTargetClass(matchedClass);
        const name = CLASS_DISPLAY_NAMES[matchedClass];
        const confirmMsg = matchedClass === 'all' 
            ? 'Đã chuyển sang nhận diện tất cả vật thể' 
            : `Đang tìm kiếm ${name}`;
        speakMessage(confirmMsg, true);
        sfx.playSuccessChime();
    } else {
        speakMessage(`Không tìm thấy vật thể "${transcript}". Vui lòng chọn: điện thoại, laptop, máy tính, hoặc máy chiếu.`, true);
    }
}

// ==========================================
// 5. Điều khiển Camera & Cập nhật Dữ liệu
// ==========================================
async function toggleCamera() {
    if (!state.isRunning) {
        // Bắt đầu camera
        try {
            DOM.btnToggleText.textContent = 'Đang kết nối...';
            const res = await fetch('/api/camera/start', { method: 'POST' });
            const data = await res.json();

            if (data.success) {
                state.isRunning = true;
                updateCameraUI(true);
                startStatusPolling();
                speakMessage('Hệ thống camera nhận diện đã hoạt động', true);
                sfx.playSuccessChime();
            } else {
                alert(data.message || 'Không thể mở webcam');
                updateCameraUI(false);
            }
        } catch (err) {
            console.error('Error starting camera:', err);
            alert('Lỗi kết nối đến máy chủ.');
            updateCameraUI(false);
        }
    } else {
        // Dừng camera
        try {
            await fetch('/api/camera/stop', { method: 'POST' });
            state.isRunning = false;
            updateCameraUI(false);
            stopStatusPolling();
            speakMessage('Đã tắt camera nhận diện', true);
        } catch (err) {
            console.error('Error stopping camera:', err);
        }
    }
}

function updateCameraUI(running) {
    if (running) {
        DOM.cameraStandby.classList.add('hidden');
        DOM.scanLine.classList.add('active');
        DOM.btnToggleCamera.classList.add('btn-danger');
        DOM.btnToggleText.textContent = 'Dừng nhận diện';
        DOM.systemStatusChip.classList.add('active');
        DOM.systemStatusText.textContent = 'Đang nhận diện';
        // Reload timestamp để trình duyệt làm mới luồng stream mjpeg nếu bị đọng cache
        DOM.cameraFeed.src = `/video_feed?t=${Date.now()}`;
    } else {
        DOM.cameraStandby.classList.remove('hidden');
        DOM.scanLine.classList.remove('active');
        DOM.btnToggleCamera.classList.remove('btn-danger');
        DOM.btnToggleText.textContent = 'Bắt đầu nhận diện';
        DOM.systemStatusChip.classList.remove('active', 'searching');
        DOM.systemStatusText.textContent = 'Sẵn sàng';
        DOM.fpsCounter.textContent = '0';
        DOM.detectedCount.textContent = '0';
        DOM.activeObjectsPill.textContent = '0 phát hiện';
        resetDirections();
    }
}

function startStatusPolling() {
    if (state.statusPollInterval) clearInterval(state.statusPollInterval);
    state.statusPollInterval = setInterval(fetchSystemStatus, 450);
}

function stopStatusPolling() {
    if (state.statusPollInterval) {
        clearInterval(state.statusPollInterval);
        state.statusPollInterval = null;
    }
}

async function fetchSystemStatus() {
    if (!state.isRunning) return;

    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Cập nhật FPS & số lượng
        DOM.fpsCounter.textContent = data.fps || 0;
        const total = data.detections ? data.detections.length : 0;
        DOM.detectedCount.textContent = total;
        DOM.activeObjectsPill.textContent = `${total} phát hiện`;

        // Cập nhật hướng di chuyển / định vị
        updateDirectionRadar(data.detections || []);

        // Cập nhật thông báo âm thanh
        if (data.latest_announcement) {
            DOM.announcementText.textContent = data.latest_announcement;
            speakMessage(data.latest_announcement);
        }

        // Cập nhật danh sách vật thể thời gian thực
        renderDetectionsList(data.detections || []);

    } catch (e) {
        console.warn('Lỗi khi fetch status:', e);
    }
}

// Cập nhật la bàn 3 hướng (Trái, Giữa, Phải)
function updateDirectionRadar(detections) {
    let leftCount = 0;
    let centerCount = 0;
    let rightCount = 0;

    detections.forEach(det => {
        if (det.pos_code === 'left') leftCount++;
        else if (det.pos_code === 'right') rightCount++;
        else centerCount++;
    });

    DOM.countLeft.textContent = leftCount;
    DOM.countCenter.textContent = centerCount;
    DOM.countRight.textContent = rightCount;

    // Toggle active highlights
    DOM.dirLeft.classList.toggle('highlight', leftCount > 0);
    DOM.dirCenter.classList.toggle('highlight', centerCount > 0);
    DOM.dirRight.classList.toggle('highlight', rightCount > 0);

    if (detections.length === 0) {
        DOM.directionBadge.textContent = 'Không có vật thể';
    } else {
        const parts = [];
        if (centerCount > 0) parts.push(`${centerCount} phía trước`);
        if (leftCount > 0) parts.push(`${leftCount} bên trái`);
        if (rightCount > 0) parts.push(`${rightCount} bên phải`);
        DOM.directionBadge.textContent = parts.join(', ');
    }
}

function resetDirections() {
    DOM.countLeft.textContent = '0';
    DOM.countCenter.textContent = '0';
    DOM.countRight.textContent = '0';
    DOM.dirLeft.classList.remove('highlight');
    DOM.dirCenter.classList.remove('highlight');
    DOM.dirRight.classList.remove('highlight');
    DOM.directionBadge.textContent = 'Chờ quét...';
    DOM.detectionsList.innerHTML = '<div class="empty-state"><p>Chưa có dữ liệu nhận diện</p></div>';
}

function renderDetectionsList(detections) {
    if (!detections || detections.length === 0) {
        DOM.detectionsList.innerHTML = '<div class="empty-state"><p>Đang quét... Không có vật thể nào trong tầm nhìn</p></div>';
        return;
    }

    const html = detections.map(item => {
        const targetClass = item.is_target ? 'target-match' : '';
        const posClass = `pos-${item.pos_code}`;
        return `
            <div class="detection-row ${targetClass}">
                <div class="det-info">
                    <span class="det-name">${item.is_target ? '⭐ ' : ''}${item.vietnamese_name}</span>
                    <span class="det-meta">Độ tin cậy: ${item.confidence}%</span>
                </div>
                <span class="det-badge ${posClass}">${item.position}</span>
            </div>
        `;
    }).join('');

    DOM.detectionsList.innerHTML = html;
}

// ==========================================
// 6. Cài đặt Mục Tiêu (Target Selection)
// ==========================================
async function setTargetClass(target) {
    state.targetClass = target;
    state.targetName = CLASS_DISPLAY_NAMES[target] || 'Tất cả';

    // Cập nhật giao diện chips
    DOM.targetChips.forEach(chip => {
        chip.classList.toggle('active', chip.dataset.target === target);
    });

    DOM.currentTargetName.textContent = state.targetName;

    // Cập nhật trạng thái badge nếu đang tìm mục tiêu
    if (target !== 'all') {
        DOM.systemStatusChip.classList.add('searching');
        DOM.systemStatusText.textContent = `Đang tìm ${state.targetName}`;
    } else {
        DOM.systemStatusChip.classList.remove('searching');
        if (state.isRunning) DOM.systemStatusText.textContent = 'Đang nhận diện';
    }

    // Gửi cập nhật đến backend
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_class: target })
        });
    } catch (err) {
        console.warn('Lỗi cập nhật target:', err);
    }
}

// ==========================================
// 7. Tải Ảnh Nhận Diện Thử Nghiệm (Upload Modal)
// ==========================================
function setupUploadModal() {
    DOM.btnUploadModal.addEventListener('click', () => {
        DOM.uploadModal.classList.add('active');
    });

    const closeModal = () => {
        DOM.uploadModal.classList.remove('active');
        DOM.fileInput.value = '';
        DOM.uploadResult.style.display = 'none';
    };

    DOM.btnCloseModal.addEventListener('click', closeModal);
    DOM.modalBackdrop.addEventListener('click', closeModal);

    // Kéo thả và chọn file
    DOM.uploadDropzone.addEventListener('click', () => DOM.fileInput.click());

    DOM.uploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        DOM.uploadDropzone.classList.add('dragover');
    });

    DOM.uploadDropzone.addEventListener('dragleave', () => {
        DOM.uploadDropzone.classList.remove('dragover');
    });

    DOM.uploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        DOM.uploadDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    DOM.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Vui lòng chọn một file ảnh hợp lệ (JPG, PNG, WEBP).');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    DOM.uploadDropzone.innerHTML = '<p>Đang phân tích hình ảnh với YOLO26n...</p>';

    try {
        const res = await fetch('/api/detect_image', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        // Khôi phục lại dropzone ban đầu
        DOM.uploadDropzone.innerHTML = `
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" x2="12" y1="3" y2="15"/>
          </svg>
          <p>Kéo và thả ảnh vào đây, hoặc <span class="browse-link">chọn từ thiết bị</span></p>
          <span class="file-hint">Hỗ trợ JPG, PNG, WEBP</span>
        `;

        if (data.success) {
            DOM.uploadResult.style.display = 'block';
            DOM.resultImage.src = data.image_data;

            if (data.detections.length === 0) {
                DOM.resultDetails.innerHTML = '<p style="color: var(--text-dim);">Không phát hiện thấy vật dụng nào trong ảnh.</p>';
                speakMessage('Không tìm thấy vật thể nào trong ảnh tải lên.', true);
            } else {
                const summary = data.detections.map(d => `${d.vietnamese_name} (${d.confidence}%, ${d.position})`).join(', ');
                DOM.resultDetails.innerHTML = `
                    <p style="color: var(--success); font-weight: 700;">Phát hiện ${data.total_detected} vật thể:</p>
                    <p style="color: var(--text-main); font-size: 0.9rem;">${summary}</p>
                `;
                speakMessage(`Ảnh có ${data.total_detected} vật thể. Phát hiện ${summary}`, true);
                sfx.playSuccessChime();
            }
        } else {
            alert(data.message || 'Lỗi khi nhận diện ảnh');
        }
    } catch (err) {
        console.error('Upload detection error:', err);
        alert('Không thể kết nối đến máy chủ.');
    }
}

// ==========================================
// 8. Event Listeners & Thiết lập ban đầu
// ==========================================
function setupEventListeners() {
    // 1. Nút Bật/Tắt Camera
    DOM.btnToggleCamera.addEventListener('click', toggleCamera);

    // 2. Nút Tìm kiếm giọng nói
    DOM.btnVoiceSearch.addEventListener('click', () => {
        if (!recognition) {
            alert('Trình duyệt của bạn không hỗ trợ Web Speech API. Vui lòng sử dụng Chrome, Edge hoặc Safari.');
            return;
        }
        if (state.isSpeechRecognitionActive) {
            recognition.stop();
        } else {
            recognition.start();
        }
    });

    // 3. Nút Bật/Tắt âm thanh
    DOM.btnToggleAudio.addEventListener('click', () => {
        state.isAudioMuted = !state.isAudioMuted;
        DOM.btnToggleAudio.classList.toggle('active', !state.isAudioMuted);
        DOM.iconAudioOn.classList.toggle('hidden', state.isAudioMuted);
        DOM.iconAudioOff.classList.toggle('hidden', !state.isAudioMuted);

        if (state.isAudioMuted) {
            if (currentSpeechAudio) {
                currentSpeechAudio.pause();
                currentSpeechAudio.currentTime = 0;
            }
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        } else {
            speakMessage('Âm thanh đã được bật', true);
            sfx.playTone(880, 0.1);
        }
    });

    // 4. Nút Đọc lại thông báo gần nhất
    DOM.btnReplayAudio.addEventListener('click', () => {
        const text = DOM.announcementText.textContent;
        if (text) speakMessage(text, true);
    });

    // 5. Chọn Chips mục tiêu
    DOM.targetChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const target = chip.dataset.target;
            setTargetClass(target);
            speakMessage(`Đã chọn tìm kiếm ${CLASS_DISPLAY_NAMES[target]}`, true);
        });
    });

    // 6. Thanh trượt Độ tin cậy (Confidence Slider)
    DOM.confSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        state.confidence = val;
        DOM.confValueDisplay.textContent = `${Math.round(val * 100)}%`;
    });

    DOM.confSlider.addEventListener('change', async (e) => {
        const val = parseFloat(e.target.value);
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ confidence: val })
        });
    });

    // 7. Giãn cách đọc giọng nói
    DOM.intervalSelect.addEventListener('change', async (e) => {
        const val = parseFloat(e.target.value);
        state.speechInterval = val;
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speech_interval: val })
        });
    });

    // 8. Chọn giọng đọc Tiếng Việt (Hoài My / Nam Minh)
    if (DOM.voiceSelect) {
        DOM.voiceSelect.addEventListener('change', (e) => {
            state.voice = e.target.value;
            const voiceLabel = e.target.options[e.target.selectedIndex].text;
            speakMessage(`Đã đổi sang ${voiceLabel}`, true);
        });
    }

    // 9. Phím tắt Trợ năng bàn phím (Keyboard Shortcuts)
    window.addEventListener('keydown', (e) => {
        // Không nhận phím tắt nếu đang gõ trong input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        // Space: Bật / Dừng camera
        if (e.code === 'Space') {
            e.preventDefault();
            toggleCamera();
        }
        // S: Tìm kiếm giọng nói
        else if (e.code === 'KeyS') {
            e.preventDefault();
            DOM.btnVoiceSearch.click();
        }
        // M: Tắt / Bật tiếng
        else if (e.code === 'KeyM') {
            e.preventDefault();
            DOM.btnToggleAudio.click();
        }
        // Escape: Đóng modal
        else if (e.code === 'Escape') {
            DOM.uploadModal.classList.remove('active');
        }
    });

    // Setup Upload Modal
    setupUploadModal();
}

// Khởi chạy khi tài liệu tải xong
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    console.log('BlindAssist Web Interface initialized successfully.');
});

