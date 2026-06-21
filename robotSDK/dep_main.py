"""
dep_main.py - 简化版（无跟踪器，无重影）
"""
import cv2
import yaml
import numpy as np
import depthai as dai
from pathlib import Path
import time
import torch
from collections import deque

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
CALIBRATION_PATH = ROOT_DIR / "stereo_calibration_results.npz"
OAK_IP = "169.254.1.222"


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "yolo": {"yolo_model": "best.pt", "yolo_conf_thresh": 0.25},
        "display_manager": {"is_display": True, "show_debug": False},
    }


def load_calibration(calib_path: Path) -> dict:
    data = np.load(calib_path)
    return {
        'K_l': data['K_l'],
        'D_l': data['D_l'],
        'K_r': data['K_r'],
        'D_r': data['D_r'],
        'R': data['R'],
        'T': data['T'],
        'baseline': abs(data['T'][0][0]),
        'fx': data['K_l'][0, 0],
        'ret_stereo': data['ret_stereo'],
        'image_size': data['image_size'],
    }


class OAKDepthSource:
    def __init__(self, calibration_path: Path, fps: int = 30, ip: str = OAK_IP):
        self.calib = load_calibration(calibration_path)
        self.ip = ip
        self.depth_width = 1280
        self.depth_height = 800
        
        print("=" * 60)
        print("OAK Depth Source (Simplified - No Tracker)")
        print("=" * 60)
        print(f"  Stereo error: {self.calib['ret_stereo']:.4f} px")
        print(f"  Baseline: {self.calib['baseline']:.2f} mm")
        print(f"  OAK IP: {self.ip}")
        
        self.pipeline = self._build_pipeline(fps)
        device_info = dai.DeviceInfo(self.ip)
        self.device = dai.Device(self.pipeline, device_info)
        
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.q_left = self.device.getOutputQueue("left", maxSize=4, blocking=False)
        self.q_right = self.device.getOutputQueue("right", maxSize=4, blocking=False)
        
        self.current_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # 异步帧队列
        self.frame_queue = deque(maxlen=2)
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
    
    def _build_pipeline(self, fps: int) -> dai.Pipeline:
        pipeline = dai.Pipeline()
        
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setFps(fps)
        mono_right.setFps(fps)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        stereo.setOutputSize(self.depth_width, self.depth_height)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setExtendedDisparity(False)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.initialConfig.setConfidenceThreshold(200)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        mono_left.out.link(xout_left.input)
        
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName("right")
        mono_right.out.link(xout_right.input)
        
        return pipeline
    
    def _reader_loop(self):
        while self.running:
            try:
                depth_data = self.q_depth.get()
                left_data = self.q_left.get()
                right_data = self.q_right.get()
                
                left_frame = left_data.getCvFrame()
                right_frame = right_data.getCvFrame()
                depth_mm = depth_data.getFrame().astype(np.float32)
                
                depth_mm_filtered = cv2.medianBlur(depth_mm.astype(np.uint16), 5).astype(np.float32)
                depth_mm_filtered[(depth_mm_filtered < 100) | (depth_mm_filtered > 5000)] = 0
                
                depth_vis = cv2.normalize(depth_mm_filtered, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                cv2.putText(depth_colored, f"FPS: {self.current_fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if len(self.frame_queue) >= 2:
                    self.frame_queue.popleft()
                self.frame_queue.append((left_frame, right_frame, depth_mm_filtered, depth_colored))
                
            except Exception as e:
                time.sleep(0.01)
    
    def get_frames(self):
        try:
            return self.frame_queue[-1] if self.frame_queue else (None, None, None, None)
        except:
            return None, None, None, None
    
    def close(self):
        self.running = False
        if hasattr(self, 'device'):
            self.device.close()


# 深度采样器（无跟踪器）
def get_object_depth(depth_map: np.ndarray, bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape
    
    x1 = max(0, min(x1, w - 2))
    y1 = max(0, min(y1, h - 2))
    x2 = max(x1 + 2, min(x2, w))
    y2 = max(y1 + 2, min(y2, h))
    
    box_h = y2 - y1
    box_w = x2 - x1
    
    regions = []
    
    # 中心区域-+
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)
    cx1 = x1 + margin_x
    cy1 = y1 + margin_y
    cx2 = x2 - margin_x
    cy2 = y2 - margin_y
    if cx2 > cx1 and cy2 > cy1:
        roi = depth_map[cy1:cy2, cx1:cx2]
        valid = roi[(roi > 100) & (roi < 5000)]
        if len(valid) > 0:
            regions.append(valid)
    
    # 上半部分
    top_roi = depth_map[y1:y1 + box_h//3, x1:x2]
    valid_top = top_roi[(top_roi > 100) & (top_roi < 5000)]
    if len(valid_top) > 5:
        regions.append(valid_top)
    
    # 下半部分
    bottom_roi = depth_map[y1 + 2*box_h//3:y2, x1:x2]
    valid_bottom = bottom_roi[(bottom_roi > 100) & (bottom_roi < 5000)]
    if len(valid_bottom) > 5:
        regions.append(valid_bottom)
    
    if len(regions) == 0:
        return 0.0, 0.0
    
    all_valid = np.concatenate(regions)
    if len(all_valid) < 5:
        return 0.0, 0.0
    
    sorted_depth = np.sort(all_valid)
    min_count = max(5, int(len(sorted_depth) * 0.15))
    depth_value = float(np.mean(sorted_depth[:min_count]))
    
    if len(sorted_depth) > 10:
        std = np.std(sorted_depth[:min_count])
        mean = np.mean(sorted_depth[:min_count])
        cv_ratio = std / (mean + 1e-6)
        confidence = max(0, min(1, 1 - cv_ratio * 2))
    else:
        confidence = 0.5
    
    return depth_value, confidence


def main():
    print("=" * 60)
    print("OAK Depth Detection (Simplified - No Tracker)")
    print("=" * 60)
    
    gpu_available = torch.cuda.is_available()
    print(f"  GPU Available: {gpu_available}")
    if gpu_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # 加载 YOLO
    from ultralytics import YOLO
    model = YOLO('best.pt')
    
    # 初始化 OAK
    depth_source = OAKDepthSource(CALIBRATION_PATH, fps=30)
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug info")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    show_debug = False
    
    while True:
        left_frame, right_frame, depth_mm, depth_colored = depth_source.get_frames()
        if left_frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # YOLO 检测（左眼）
        left_rgb = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
        results = model(left_rgb, conf=0.3)
        
        left_boxes = []
        for r in results:
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    left_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': float(conf[i]),
                        'class_name': model.names[int(cls[i])]
                    })
        
        # 左眼显示
        left_display = left_rgb.copy()
        for box in left_boxes:
            bbox = (box['x1'], box['y1'], box['x2'], box['y2'])
            raw_depth, confidence = get_object_depth(depth_mm, bbox)
            
            cv2.rectangle(left_display, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
            cv2.putText(left_display, f"{box['class_name']} {box['conf']:.2f}", 
                        (box['x1'], box['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if raw_depth > 0:
                cv2.putText(left_display, f"{raw_depth:.0f}mm", 
                            (box['x1'], box['y2'] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 深度图显示
        depth_display = depth_colored.copy()
        for box in left_boxes:
            bbox = (box['x1'], box['y1'], box['x2'], box['y2'])
            raw_depth, confidence = get_object_depth(depth_mm, bbox)
            cv2.rectangle(depth_display, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 255), 2)
            if raw_depth > 0:
                cv2.putText(depth_display, f"{raw_depth:.0f}mm", 
                            (box['x1'], box['y2'] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 并排显示
        h, w = left_display.shape[:2]
        depth_resized = cv2.resize(depth_display, (w, h))
        combined = np.hstack([left_display, depth_resized])
        
        cv2.putText(combined, "Left + YOLO (No Tracker)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Depth Map", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # FPS
        if time.time() - start_time > 1.0:
            fps_display = frame_count / (time.time() - start_time)
            cv2.putText(combined, f"FPS: {fps_display:.1f}", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame_count = 0
            start_time = time.time()
        
        cv2.namedWindow('OAK Detection (No Tracker)', cv2.WINDOW_NORMAL)
        cv2.imshow('OAK Detection (No Tracker)', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
    
    depth_source.close()
    cv2.destroyAllWindows()
    print("已退出 ✅")


if __name__ == "__main__":
    import threading
    main()