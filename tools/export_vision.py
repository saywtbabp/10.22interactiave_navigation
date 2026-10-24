"""CPU export/held-out evaluation using existing Python 3.9 + local wheels."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.training_deps38'))
sys.path.insert(0, str(ROOT/'.training_deps'))  # CPython 3.9 ONNX wheel first.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT/'.matplotlib')
os.environ['YOLO_AUTOINSTALL'] = 'false'
import json
import hashlib
import shutil
import argparse


def main():
    import onnx
    import torch
    from ultralytics import YOLO
    p = argparse.ArgumentParser(); p.add_argument('--epochs', type=int, default=35); a = p.parse_args()
    torch.set_num_threads(4)
    checkpoint = ROOT/'results'/'vision_training'/'door_glass'/'weights'/'best.pt'
    model = YOLO(str(checkpoint))
    metrics_path = ROOT/'results'/'vision_training'/'test_metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    exported = model.export(format='onnx', imgsz=320, opset=12, simplify=False, dynamic=False, device='cpu')
    folder = ROOT/'models'/'vision'; dest = folder/'door_glass_yolov8n.onnx'
    shutil.copy2(exported, dest); shutil.copy2(checkpoint, folder/'door_glass_yolov8n.pt')
    onnx.checker.check_model(onnx.load(str(dest)))
    report = dict(architecture='Ultralytics YOLOv8n', ultralytics_version='8.3.40',
                  classes={'0':'door','1':'glass'}, input_size=320, epochs=a.epochs,
                  onnx_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  test_metrics=metrics,
                  training_source='data/vision_synthetic/manifest.json',
                  scope='synthetic MuJoCo domain only; scores are not material or movability probabilities',
                  license='AGPL-3.0, https://ultralytics.com/license')
    (folder/'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
