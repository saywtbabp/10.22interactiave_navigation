"""Evaluate the DELIVERED ONNX pipeline on held-out synthetic images at IoU .5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from configuration import ROOT, load_config
from navigation.vision import YoloDetector, iou


def main():
    cfg = load_config(); detector = YoloDetector(cfg)
    counts = {k:dict(tp=0, fp=0, fn=0) for k in ('door','suspected_glass')}; times=[]
    data = ROOT/'data'/'vision_synthetic'
    for image in sorted((data/'images'/'test').glob('*.jpg')):
        rgb = np.asarray(Image.open(image).convert('RGB')); h,w = rgb.shape[:2]
        frame = detector.infer(SimpleNamespace(rgb=rgb)); times.append(frame.inference_ms)
        labels = []
        for line in (data/'labels'/'test'/(image.stem+'.txt')).read_text().splitlines():
            cls,cx,cy,bw,bh = map(float,line.split())
            labels.append(('door' if cls==0 else 'suspected_glass',
                           [(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        used=set()
        for d in frame.detections:
            choices=[(iou(d.bbox_xyxy,b),j) for j,(kind,b) in enumerate(labels) if j not in used and kind==d.class_name]
            best=max(choices,default=(0.,-1))
            if best[0]>=.5:used.add(best[1]);counts[d.class_name]['tp']+=1
            else:counts[d.class_name]['fp']+=1
        for j,(kind,_) in enumerate(labels):
            if j not in used:counts[kind]['fn']+=1
    for c in counts.values():
        c['precision']=c['tp']/max(1,c['tp']+c['fp']);c['recall']=c['tp']/max(1,c['tp']+c['fn'])
    report=dict(model_sha256=detector.sha,images=len(times),iou=.5,confidence=cfg.yolo_confidence,
                counts=counts,inference_ms_mean=float(np.mean(times)),inference_ms_p95=float(np.percentile(times,95)),
                scope='independent randomized synthetic scene instances, not a real-world benchmark')
    out=ROOT/'results'/'sensor_upgrade_validation';out.mkdir(exist_ok=True)
    (out/'onnx_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    assert all(c['precision']>.8 and c['recall']>.8 for c in counts.values()),report


if __name__=='__main__':main()
