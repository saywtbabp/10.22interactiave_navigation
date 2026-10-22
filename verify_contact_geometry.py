"""Regression from the saved resistant-door failure, without shrinking clearance."""
import runtime
import numpy as np
from configuration import ROOT,Config,architecture
from navigation.contact_geometry import contact_comparison
from navigation.interactions import merge_pose
from navigation.geometry import Planner
from experiments.library import save_json

def main():
    a=architecture();cfg=Config();o=a['openings'][0]
    start=np.array([.7532211255154141,2.4042260122967245,1.2924741193986713])
    result=contact_comparison(a,cfg,o,start,merge_pose(o))
    assert cfg.clearance==.34
    assert 0<result['initial_inflation_overlap_m']<.003
    assert result['monotone_contact_egress_connector_m']<.01
    assert result['G']>3
    # All ordinary A* nodes after the one declared initial connector remain safe.
    planner=Planner(a,cfg,sealed=['O02'],row=2.6)
    assert all(planner.clear(p,q) for p,q in zip(result['door_path'][1:-1],result['door_path'][2:]))
    save_json(ROOT/'results'/'contact_geometry_regression.json',result)
    print('PASS: measured contact state exits {:.6f} m inflation overlap; G={:.6f} m'.format(result['initial_inflation_overlap_m'],result['G']))

if __name__=='__main__':main()
