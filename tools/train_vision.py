"""Offline YOLOv8n fine tuning and ONNX export; normal main.py needs no torch.

Use the existing CUDA Python 3.8 environment; project-local wheels only.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.training_deps'))
sys.path.insert(0, str(ROOT/'.training_deps38'))
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT/'.matplotlib')
os.environ['WANDB_MODE'] = 'disabled'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '4'
import json
import shutil
import hashlib
import argparse
import subprocess


def main():
    from ultralytics import YOLO
    import torch
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--export-python', default='G:/anaconda3/envs/py39/python.exe')
    a = p.parse_args()
    torch.set_num_threads(4)
    model = YOLO(str(ROOT/'models'/'vision'/'yolov8n.pt'))
    model.train(data=str(ROOT/'data'/'vision_synthetic'/'data.yaml'), epochs=a.epochs, imgsz=320,
                batch=16, device=0 if torch.cuda.is_available() else 'cpu', workers=0, seed=1847,
                project=str(ROOT/'results'/'vision_training'), name='door_glass', exist_ok=True,
                pretrained=True, patience=12, amp=False, plots=True, save=True,
                hsv_h=.08, hsv_s=.7, hsv_v=.4, fliplr=.5, mosaic=.5, close_mosaic=5)
    best = ROOT/'results'/'vision_training'/'door_glass'/'weights'/'best.pt'
    trained = YOLO(str(best))
    metrics = trained.val(data=str(ROOT/'data'/'vision_synthetic'/'data.yaml'), split='test', imgsz=320,
                          device=0 if torch.cuda.is_available() else 'cpu', workers=0, plots=True)
    (ROOT/'results'/'vision_training'/'test_metrics.json').write_text(
        json.dumps({k:float(v) for k,v in metrics.results_dict.items()},indent=2),encoding='utf-8')
    # Python 3.8.0's ONNX DLL fails to initialize on this host. Export in the
    # existing Python 3.9 CPU environment; neither Conda installation is edited.
    subprocess.run([a.export_python, str(ROOT/'tools'/'export_vision.py'), '--epochs', str(a.epochs)], check=True)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support(); main()
