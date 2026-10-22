# -*- coding: utf-8 -*-
"""Final evidence audit; reads completed runs, never fabricates task outcomes."""
import runtime
import json
import hashlib
from pathlib import Path
import numpy as np
from configuration import ROOT,load_config,architecture
from experiments.audit import audit
from experiments.library import signature,save_json

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    root=ROOT/'results';cfg=load_config();arch=architecture();opening=arch['openings'][0]
    selected={'standard':root/'20261022_194714_standard_adaptive',
              'resistant':root/'20261022_194101_standard_adaptive'}
    checks={};details={}
    for name,folder in selected.items():
        summary=read(folder/'summary.json');decisions=read(folder/'decisions.json');a=audit(folder)
        assert summary['success'] and summary['config']['sobol_power']==10
        assert a['motor_limits_pass'] and a['fixed_height_pass'] and a['no_wall_contact']
        assert a['glass_contacts']['O03']==1
        assert a['observed_records']['O07']['traversability']=='free'
        assert [d['action'] for d in decisions]==(['Probe','Push','Avoid'] if name=='standard' else ['Probe','Avoid','Avoid'])
        beliefs=read(folder/'door_beliefs.json')
        assert len(beliefs['O01']['weights'])==1024 and abs(sum(beliefs['O01']['weights'])-1)<1e-12
        assert np.allclose(beliefs['O05']['weights'],1/1024,rtol=0,atol=1e-15)
        weights=np.array(beliefs['O01']['weights']);labels=np.array(beliefs['O01']['labels'])
        assert abs(float(weights@labels)-decisions[1]['p_D'])<1e-12
        details[name]=dict(folder=folder.relative_to(ROOT).as_posix(),summary=summary,audit=a,
                           first_action=decisions[0]['action'],initial_p_D=decisions[0]['p_D'],initial_D=decisions[0]['D'],
                           post_action=decisions[1]['action'],post_p_D=decisions[1]['p_D'],post_D=decisions[1]['D'])
    checks['unobserved_truth_does_not_change_initial_action']=all(details['standard'][k]==details['resistant'][k] for k in ('first_action','initial_p_D','initial_D'))
    caches=[]
    for directory in ('O01_initial_09c40fbb7dcbb3a5','O01_post_probe_08a6737c610b1222','O01_post_probe_2296140ce356374b'):
        folder=root/'cache'/directory;meta=read(folder/'metadata.json')
        assert meta['N']==1024
        expected=signature(arch,cfg,opening,meta['observed_initial_state'],meta['mode'])
        assert expected==meta['signature']
        with np.load(folder/'worlds.npz',allow_pickle=False) as a:
            assert a['done'].all() and len(a['theta'])==1024 and str(a['signature'])==expected
            assert not np.isnan(a['J_door']).any() and not np.isnan(a['J_bypass']).any()
            assert not any('numerical_anomaly' in s for s in a['door_reason'])
            if meta['mode']=='initial':assert np.isfinite(a['stream']).all()
        caches.append(dict(directory=directory,signature=expected,N=1024,complete=True))
    for case in ('standard','all_empty'):
        live=read(root/('gui_consistency_'+case)/'checks.json');assert all(live.values())
        checks['gui_'+case]=live
    imu=read(selected['standard']/'imu_measurements.json');assert len(imu)>1000
    assert all(np.isfinite(x['imu_acc_m_s2']).all() for x in imu)
    checks['actual_imu_records']=len(imu)
    checks['runtime_cache_and_memory']=read(root/'runtime_verification'/'checks.json')
    checks['degraded_sensor_navigation']=read(root/'degraded_sensing_c025_r025'/'degraded_checks.json')
    assert checks['degraded_sensor_navigation']['success']
    manifest={}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or 'results' in path.relative_to(ROOT).parts or '__pycache__' in path.parts:continue
        if path.suffix in ('.py','.xml','.json','.png','.txt','.md','.svg'):
            manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    original=ROOT.parent/'Mujoco_0'/'tensegrity_real_model.xml'
    checks['original_model_copy_identical']=original.read_bytes()==(ROOT/'assets'/'models'/'tensegrity_real_model.xml').read_bytes()
    assert checks['original_model_copy_identical']
    for path in ROOT.rglob('*.py'):
        if 'results' not in path.relative_to(ROOT).parts:compile(path.read_bytes(),str(path),'exec')
    save_json(root/'release_validation.json',dict(checks=checks,runs=details,current_caches=caches,source_sha256=manifest))
    print('PASS: final full missions, independent beliefs, 3 current 1024 caches, real GUI equality, IMU and source integrity',flush=True)

if __name__=='__main__':main()
