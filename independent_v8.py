import cv2
import threading
import time
from ultralytics import YOLO
import numpy as np
import math
import psutil
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
    TEST_DURATION = 60  # 測試秒數 
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
    
    window_name = "independent v8 Inference"
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

        grid_rows = []
        for r in range(rows):
            row_frames = processed_frames[r * cols : (r + 1) * cols]
            grid_rows.append(cv2.hconcat(row_frames))
        super_frame = cv2.vconcat(grid_rows)
        
        infer_start_time = time.time()
        results = model(super_frame, conf=0.30, verbose=False)
        loop_infer_time = time.time() - infer_start_time
        
        annotated_super_frame = results[0].plot()

        for i, f in enumerate(batch_frames):
            if f is None:
                col_idx = i % cols   
                row_idx = i // cols  
                offset_x = col_idx * CELL_W
                offset_y = row_idx * CELL_H
                
                text = "Streaming not found"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
                text_x = offset_x + (CELL_W - text_size[0]) // 2
                text_y = offset_y + (CELL_H + text_size[1]) // 2
                cv2.putText(annotated_super_frame, text, (text_x, text_y), font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

        elapsed = time.time() - start_time
        cv2.putText(annotated_super_frame, f"Plan B Testing... {int(TEST_DURATION - elapsed)}s left", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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

    streamer.stop()
    cv2.destroyAllWindows()
    
    real_duration = time.time() - start_time
    
    if total_frames_processed == 0:
        exit()

    avg_cpu = np.mean(cpu_usages)
    avg_ram = np.mean(ram_usages)
    
    avg_infer_latency_ms = (total_inference_time / total_frames_processed) * 1000
    avg_e2e_latency_ms = (total_e2e_time / total_frames_processed) * 1000
    
    total_input_fps = (total_frames_processed * num_cams) / real_duration
    single_channel_fps = total_frames_processed / real_duration

    print("\n\n" + "="*60)
    print("="*60)
    print(f"測試完成時間：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 實際測試時長：{real_duration:.1f} 秒")
    print(f" 總執行迴圈數：{total_frames_processed} 圈\n")
    
    print(f"總輸入FPS:     {total_input_fps:.2f}")
    print(f" 單路平均FPS:   {single_channel_fps:.2f}")
    print(f" 推論延遲(ms):  {avg_infer_latency_ms:.2f} ")
    print(f" 端到端延遲(ms): {avg_e2e_latency_ms:.2f} ")
    print(f" CPU平均(%):    {avg_cpu:.1f}%")
    print(f" RAM平均(GB):   {avg_ram:.2f} GB")
    
    if HAS_GPUTIL and gpu_usages:
        avg_gpu = np.mean(gpu_usages)
        avg_vram = np.mean(vram_usages)
        print(f" GPU平均(%):    {avg_gpu:.1f}%")
        print(f" VRAM平均(GB):  {avg_vram:.2f} GB")
    else:
        print(" GPU平均(%):    N/A ")
        print(" VRAM平均(GB):  N/A")
    print("="*60)