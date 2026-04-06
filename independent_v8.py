import cv2
import threading
import time
from ultralytics import YOLO
import numpy as np
import math
import psutil
import subprocess

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

class MultiCamStream:
    def __init__(self, urls):
        self.urls = urls
        self.num_cams = len(urls)
        self.frames = [None] * self.num_cams
        self.grabbed_counts = [0] * self.num_cams  # 記錄每路相機實際抓取到的幀數
        self.stopped = False
        self.threads = []
        print(f"啟動 {self.num_cams} 支相機的背景拉流")
        for i, url in enumerate(self.urls):
            t = threading.Thread(target=self._update, args=(i, url), daemon=True)
            t.start()
            self.threads.append(t)

    def _update(self, index, url):
        cap = cv2.VideoCapture(url)
        is_video_file = isinstance(url, str) and url.endswith('.mp4')
        while not self.stopped:
            if not cap.isOpened():
                self.frames[index] = None 
                time.sleep(1)
                cap = cv2.VideoCapture(url)
                continue
            ret, frame = cap.read()
            if ret:
                self.frames[index] = frame
                self.grabbed_counts[index] += 1  # 成功抓取一幀就 +1
                if is_video_file: time.sleep(0.033) 
            else:
                self.frames[index] = None 
                if is_video_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: time.sleep(0.01)
        cap.release()

    def read(self): return self.frames.copy()
    def stop(self):
        self.stopped = True
        for t in self.threads: t.join() 

def get_gpu_power():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        return float(result.strip().split('\n')[0])
    except Exception:
        return 0.0

def letterbox_image(img, expected_size):
    ih, iw = img.shape[0:2]
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(img, (nw, nh))
    new_image = np.zeros((eh, ew, 3), np.uint8)
    top, left = (eh - nh) // 2, (ew - nw) // 2
    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image

if __name__ == '__main__':
    TEST_DURATION = 300  # 測試秒數 
    MODEL_WEIGHTS = "yolov8n.pt"
    CELL_W, CELL_H = 640, 360    # 單路解析度
    
    model = YOLO(MODEL_WEIGHTS) 
    
    camera_urls = [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4", 
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",     
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4", 
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4" 
    ] 
    
    streamer = MultiCamStream(camera_urls)
    time.sleep(2) # 等待攝影機連線
    
    window_name = "Independent Inference v8"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720) 

    # --- 效能監控變數初始化 ---
    total_frames_processed = 0
    total_inference_time = 0
    total_e2e_time = 0
    cpu_usages = []
    ram_usages = []
    gpu_usages = []
    vram_usages = []
    power_usages = []
    
    num_cams = len(camera_urls)
    cols = math.ceil(math.sqrt(num_cams)) 
    rows = math.ceil(num_cams / cols)

    start_time = time.time()

    while (time.time() - start_time) < TEST_DURATION:
        loop_start_time = time.time()

        batch_frames = streamer.read()
        
        valid_frame = next((f for f in batch_frames if f is not None), None)
        if valid_frame is None:
            time.sleep(0.1)
            continue
            
        black_frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        processed_frames = []

        infer_start_time = time.time()
        
        for i, f in enumerate(batch_frames):
            if f is not None:
                # 先把原始畫面縮放到指定大小
                resized_f = letterbox_image(f, (CELL_W, CELL_H))
                # 獨立送進模型運算！
                results = model(resized_f, conf=0.30, verbose=False)
                # 取得畫好框框的圖，存入準備拼圖的陣列
                processed_frames.append(results[0].plot())
            else:
                blank = black_frame.copy()
                text = "Streaming not found"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
                text_x = (CELL_W - text_size[0]) // 2
                text_y = (CELL_H + text_size[1]) // 2
                cv2.putText(blank, text, (text_x, text_y), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                processed_frames.append(blank)
                
        loop_infer_time = time.time() - infer_start_time
        # ==========================================
                
        # 補齊空缺的格子 (如果攝影機數量不是完美矩形)
        while len(processed_frames) < (rows * cols):
            processed_frames.append(black_frame.copy())

        # 開始拼圖顯示 (把已經畫好框的 4 張小圖拼起來)
        grid_rows = []
        for r in range(rows):
            row_frames = processed_frames[r * cols : (r + 1) * cols]
            grid_rows.append(cv2.hconcat(row_frames))
        annotated_super_frame = cv2.vconcat(grid_rows)

        elapsed = time.time() - start_time
        cv2.putText(annotated_super_frame, f" Independent Testing... {int(TEST_DURATION - elapsed)}s left", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, annotated_super_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        loop_e2e_time = time.time() - loop_start_time
        
        total_inference_time += loop_infer_time
        total_e2e_time += loop_e2e_time
        total_frames_processed += 1 
        
        cpu_usages.append(psutil.cpu_percent(interval=None))
        ram_usages.append(psutil.virtual_memory().used / (1024**3))
        
        if HAS_GPUTIL:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usages.append(gpus[0].load * 100)
                vram_usages.append(gpus[0].memoryUsed / 1024)
                power_usages.append(get_gpu_power()) # 記錄當下功耗

    streamer.stop()
    cv2.destroyAllWindows()
    
    real_duration = time.time() - start_time
    
    if total_frames_processed == 0:
        exit()

    # 計算掉幀率
    total_grabbed = sum(streamer.grabbed_counts)
    total_processed_cams = total_frames_processed * num_cams
    if total_grabbed > 0:
        drop_rate = max(0.0, ((total_grabbed - total_processed_cams) / total_grabbed) * 100)
    else:
        drop_rate = 0.0

    print("\n" + "="*50)
    print("="*50)
    print(f"測試時長: {real_duration:.1f} 秒")
    print(f"總處理循環數: {total_frames_processed} 圈")
    print(f"總輸入FPS:     {(total_frames_processed * num_cams) / real_duration:.2f}")
    print(f"單路平均FPS:   {total_frames_processed / real_duration:.2f}")
    print(f"推論延遲(ms):  {(total_inference_time / total_frames_processed) * 1000:.2f}")
    print(f"端到端延遲(ms): {(total_e2e_time / total_frames_processed) * 1000:.2f}")
    print(f"CPU 平均(%):   {np.mean(cpu_usages):.1f}%")
    print(f"RAM 平均(GB):  {np.mean(ram_usages):.2f} GB")
    
    if HAS_GPUTIL and gpu_usages:
        print(f"GPU 平均(%):   {np.mean(gpu_usages):.1f}%")
        print(f"VRAM 平均(GB): {np.mean(vram_usages):.2f} GB")
        print(f"功耗平均(W):   {np.mean(power_usages):.1f} W")
    else:
        print("GPU 平均(%):   N/A")
        print("VRAM 平均(GB): N/A")
        print("功耗平均(W):   N/A")
        
    print(f"掉幀率(%):     {drop_rate:.1f}%")
    print("="*50)
