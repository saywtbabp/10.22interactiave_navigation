"""Audit real output folders without turning absent tests into passes."""
import csv
import json
from pathlib import Path
import numpy as np
from .library import save_json

def audit(folder):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    with (folder/'flight.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    motors=np.array([[float(r['motor'+str(i)+'_N']) for i in range(1,5)] for r in rows])
    mapdata=json.loads((folder/'online_map.json').read_text(encoding='utf-8'))
    events=json.loads((folder/'events.json').read_text(encoding='utf-8'))
    audit=dict(success=summary['success'],motor_range_N=[float(motors.min()),float(motors.max())],
               motor_limits_pass=bool((motors>=-1e-10).all() and (motors<=10+1e-10).all()),
               fixed_height_pass=summary['height_range_m'][0]>.5 and summary['height_range_m'][1]<2.05,
               no_wall_contact=summary['wall_contact_peak_N']<.1,
               glass_contacts={oid:sum(h['outcome']=='blocked' for h in r['contact_history']) for oid,r in mapdata['records'].items() if r['semantic']=='suspected_glass'},
               glass_verifications={oid:len(r['contact_history']) for oid,r in mapdata['records'].items() if r['semantic']=='suspected_glass'},
               observed_records={oid:dict(semantic=r['semantic'],traversability=r['traversability'],bypass=r['bypass_selected'])
                                 for oid,r in mapdata['records'].items()},
               map_updates=len(events),frame_count=summary['sensor_frames'])
    truth=folder/'evaluation_truth.json'
    if truth.exists():
        records=json.loads(truth.read_text(encoding='utf-8'));errors=[]
        for record in records:
            for door in record['doors'].values():
                age=door['estimate_age_s']
                if age is not None and 0<=age<=.15:
                    errors.append(abs(door['estimated_angle_rad']-door['truth_angle_rad']))
        if errors:audit['valid_age_angle_error_deg']=dict(samples=len(errors),median=float(np.degrees(np.median(errors))),p95=float(np.degrees(np.quantile(errors,.95))),max=float(np.degrees(max(errors))))
    save_json(folder/'audit.json',audit);return audit
