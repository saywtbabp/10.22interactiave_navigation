"""Held-out evaluation using the working existing GPU torchvision installation."""
from train_vision import ROOT
import json


if __name__ == '__main__':
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    model = YOLO(str(ROOT/'results'/'vision_training'/'door_glass'/'weights'/'best.pt'))
    metrics = model.val(data=str(ROOT/'data'/'vision_synthetic'/'data.yaml'), split='test', imgsz=320,
                        device=0, workers=0, plots=True, project=str(ROOT/'results'/'vision_training'), name='test_gpu')
    (ROOT/'results'/'vision_training'/'test_metrics.json').write_text(
        json.dumps({k:float(v) for k,v in metrics.results_dict.items()},indent=2),encoding='utf-8')
