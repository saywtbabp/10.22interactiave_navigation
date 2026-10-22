"""Cancellation/resume, invalid-cache rejection and evidence lifecycle tests.

32-world construction below is an explicitly diagnostic fixture; the formal
default mission separately uses the full 1024 candidates.
"""
import runtime
from copy import deepcopy
from dataclasses import replace
from multiprocessing import freeze_support
import io
import time
import numpy as np
from configuration import ROOT,Config,architecture
from navigation.physics import World
from navigation.sensors import Sensors
from navigation.perception import Memory
from navigation.interactions import approach,observed_mechanical_state
from experiments.library import build,signature,save_json

class DiagnosticCancel(Exception):pass

def main():
    cfg=replace(Config(),sobol_power=5,workers=2,seed=501);arch=architecture();o=arch['openings'][0]
    test_arch=deepcopy(arch);test_arch['start_xyz_m']=approach(o).tolist()
    w=World(test_arch,cfg);w.hold(approach(o),.6,yaw=0.)
    state=observed_mechanical_state(w,o,lambda:(0.,0.));checks={}
    key=signature(arch,cfg,o,state,'initial')
    checks['domain_change_invalidates']=key!=signature(arch,replace(cfg,friction_max=7.9),o,state,'initial')
    checks['observation_scale_invalidates']=key!=signature(arch,replace(cfg,force_scale=.31),o,state,'initial')
    checks['worker_count_does_not_change_physics_cache']=key==signature(arch,replace(cfg,workers=3),o,state,'initial')
    altered=deepcopy(state);altered['panel_angle_rad']=.01
    checks['post_state_invalidates']=key!=signature(arch,cfg,o,altered,'initial')
    started=time.perf_counter();count=[0]
    def stop(done,total):
        count[0]+=1
        if count[0]>=3:raise DiagnosticCancel('intentional cancellation fixture')
    try:build(arch,cfg,o,state,progress=stop)
    except DiagnosticCancel:checks['cancelled']=True
    checks['cancel_seconds']=time.perf_counter()-started
    lib=build(arch,cfg,o,state);checks['resumed_all_32_diagnostic_worlds']=bool(lib['done'].all())
    file=lib['folder']/'worlds.npz';original=file.read_bytes()
    fixture=ROOT/'results'/'runtime_verification';fixture.mkdir(exist_ok=True)
    with np.load(io.BytesIO(original),allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    arrays['door_reason'][0]='numerical_anomaly_diagnostic_fixture'
    try:
        with file.open('wb') as f:np.savez_compressed(f,**arrays)
        (fixture/'intentionally_invalid_cache.npz').write_bytes(file.read_bytes())
        try:build(arch,cfg,o,state)
        except RuntimeError as e:checks['invalid_completed_cache_rejected']='numerical-anomaly' in str(e)
    finally:file.write_bytes(original)
    # Real RGB for an empty aperture and then a newly present door at that same
    # location: persistent free memory must trigger explicit re-verification.
    packets=[]
    for variant in ('all_empty','standard'):
        world=World(test_arch,cfg,variant=variant);s=Sensors(world,cfg)
        for i in range(7):
            world.hold(approach(o),.1,yaw=0.);packet,_=s.acquire(force=True);packet=deepcopy(packet)
            packet.time=(0. if variant=='all_empty' else 1.)+i*.1;packets.append(packet)
        s.close()
    memory=Memory(arch,cfg)
    for packet in packets:memory.fuse(packet)
    checks['contradiction_explicitly_reverified']=any(e['event']=='contradiction_requires_reverification' for e in memory.events)
    checks['new_door_not_free']=memory.records[o['id']].traversability!='free'
    # Two old frames cannot be combined with one fresh frame after a gap.
    gap=Memory(arch,cfg)
    for packet in packets[:2]:gap.fuse(packet)
    third=deepcopy(packets[2]);third.time=2.;gap.fuse(third)
    checks['stale_candidate_prefix_not_counted']=gap.records[o['id']].traversability=='unverified'
    save_json(fixture/'checks.json',checks);print(checks,flush=True)
    assert all(v for k,v in checks.items() if k!='cancel_seconds'),checks
    assert checks['cancel_seconds']<15,checks

if __name__=='__main__':freeze_support();main()
