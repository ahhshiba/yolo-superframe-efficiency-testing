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
        self.grabbed_counts = [0] * self.num_cams  # 新增：記錄每支相機實際抓取到的幀數
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
                self.grabbed_counts[index] += 1  # 新增：成功抓取一幀就 +1
                if is_video_file: time.sleep(0.033) 
            else:
                self.frames[index] = None 
                if is_video_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else: time.sleep(0.01)
        cap.release()

    def read(self): 
        return self.frames.copy()
        
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
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_resized = cv2.resize(img, (nw, nh))
    new_image = np.zeros((eh, ew, 3), np.uint8)
    
    top = (eh - nh) // 2
    left = (ew - nw) // 2
    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    
    camera_urls = [
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4", 
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",     
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4", 
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4" 
    ] 
    streamer = MultiCamStream(camera_urls)
    time.sleep(2) 
    
    CELL_W, CELL_H = 640, 360
    window_name = "Super Frame Inference v8"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720) 

    TEST_DURATION = 300
    total_frames_processed = 0
    total_inference_time = 0
    total_e2e_time = 0
    cpu_usages, ram_usages, gpu_usages, vram_usages = [], [], [], []
    power_usages = [] # 新增：記錄功耗
    
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
        for f in batch_frames:
            if f is not None: 
                processed_frames.append(letterbox_image(f, (CELL_W, CELL_H)))
            else: 
                processed_frames.append(black_frame.copy())

        while len(processed_frames) < (rows * cols):
            processed_frames.append(black_frame.copy())

        # 拼圖v8
        grid_rows = []
        for r in range(rows):
            row_frames = processed_frames[r * cols : (r + 1) * cols]
            grid_rows.append(cv2.hconcat(row_frames))
        super_frame = cv2.vconcat(grid_rows)
        
        infer_start = time.time()
        results = model(super_frame, conf=0.30, verbose=False)
        infer_time = time.time() - infer_start
        
        annotated_super_frame = results[0].plot()

        for i, f in enumerate(batch_frames):
            if f is None:
                offset_x = (i % cols) * CELL_W
                offset_y = (i // cols) * CELL_H
                cv2.putText(annotated_super_frame, "Streaming not found", (offset_x+100, offset_y+200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # countdown
        elapsed = time.time() - start_time
        cv2.putText(annotated_super_frame, f" Testing... {int(TEST_DURATION - elapsed)}s left", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, annotated_super_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

        loop_time = time.time() - loop_start_time
        total_inference_time += infer_time
        total_e2e_time += loop_time
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
