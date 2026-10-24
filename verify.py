"""Meaningful geometry, physical mapping, sensing and belief checks.

This entry is separate from complete-task runs; its report does not claim flight
branches were executed merely because a unit or static geometry test passed.
"""
import runtime
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import numpy as np
import mujoco
from configuration import ROOT,Config,architecture
from navigation.physics import World
from navigation.sensors import Sensors
from navigation.perception import Memory
from navigation.geometry import Planner,door_comparison
from navigation.interactions import approach,merge_pose,panel_geometry,observed_mechanical_state,initialize_prediction
from navigation.sampling import candidate_worlds
from navigation.belief import consistency_update,support,decide
from experiments.library import save_json

def run():
    cfg=Config();arch=architecture();report={};theta=candidate_worlds(cfg)
    assert theta.shape==(1024,2) and len(np.unique(theta,axis=0))==1024
    w=np.ones(1024)/1024;labels=(theta[:,1]<4).astype(int)
    p,D=support(w,labels);assert abs(D-4*p*(1-p))<1e-15
    predictions=np.column_stack((theta[:,1],theta[:,0]));posterior,_=consistency_update(w,predictions,np.array([.01,.05]),cfg)
    assert abs(posterior.sum()-1)<1e-12 and posterior.argmax()==0
    assert decide(2,p,D,cfg)[0]=='Avoid' and decide(5,p,D,cfg)[0]=='Probe'
    assert decide(5,.99,.0396,cfg)[0]=='Push' and decide(5,.01,.0396,cfg)[0]=='Avoid'
    assert decide(5,.5,1,cfg,probe_used=True)[0]=='Avoid'
    report['candidate_math']=dict(N=1024,weights_sum=float(posterior.sum()),uniform_p=p,D=D,all_gate_branches=True)
    assert arch['openings'][0]['center_xy_m']==[.5,2.6]
    assert arch['openings'][5]['center_xy_m']==[3.7,8.2]
    geometry=[]
    for radius in (.34,.40):
        for a in (0.,math.radians(75)):
            for b in (0.,math.radians(75)):
                for glass in (False,True):
                    panels=[panel_geometry(arch['openings'][0],a)[2],panel_geometry(arch['openings'][4],b)[2]]
                    if glass:panels.append((np.array([-1.5,5.4]),np.eye(2),np.array([.635,.018])))
                    planner=Planner(arch,replace(cfg,clearance=radius),panels=panels)
                    path,length=planner.plan(arch['start_xyz_m'],arch['goal_xyz_m'])
                    assert all(planner.clear(p,q) for p,q in zip(path[:-1],path[1:]))
                    geometry.append(dict(radius=radius,door_angles_rad=[a,b],glass=glass,length_m=length,clear=True))
    report['geometry_16']=geometry
    report['geometric_gain']={}
    for i in (0,4):
        o=arch['openings'][i];g=door_comparison(arch,cfg,o,approach(o),merge_pose(o));report['geometric_gain'][o['id']]=g
    inertia=[]
    for I in (.05,.317,.80):
        world=World(arch,cfg,door_parameters={'O01':(I,.035),'O05':(.437,2.473)},visual=False)
        j=world.model.joint('hinge_O01').id;da=int(world.model.jnt_dofadr[j]);M=np.zeros((world.model.nv,world.model.nv))
        mujoco.mj_fullM(world.model,M,world.data.qM);assert abs(M[da,da]-I)<1e-10
        assert world.model.nu==7 and len(world.motor_ids)==4
        assert world.model.ntendon==24
        inertia.append(dict(requested_I=I,mass_matrix_I=float(M[da,da]),tendons=world.model.ntendon))
    report['physical_inertia']=inertia
    # Real sensor images in content/position variants, legal different initial poses.
    sensor_cases=[]
    for variant,index,expected in [('standard',0,'door'),('standard',2,'suspected_glass'),('standard',3,'pending_verification'),
                                  ('glass_removed',2,'pending_verification'),('swapped',0,'suspected_glass'),('swapped',2,'door')]:
        o=arch['openings'][index];test_arch=deepcopy(arch);test_arch['start_xyz_m']=approach(o).tolist()
        world=World(test_arch,cfg,variant=variant);s=Sensors(world,cfg);memory=Memory(arch,cfg)
        def sample(world):
            packet,fresh=s.acquire()
            if fresh:memory.fuse(packet)
        world.callback=sample;world.hold(approach(o),.75,yaw=0.)
        rec=memory.records[o['id']];actual='pending_verification' if rec.semantic=='unknown' and rec.verification_state=='pending' else rec.semantic
        assert actual==expected,(variant,o['id'],actual,expected,rec.visual_evidence,rec.range_evidence)
        before=memory.version;assert not memory.fuse(s.packet) and memory.version==before
        # Missing RGB and all invalid ranges must not turn unknown into free.
        invalid=deepcopy(s.packet);invalid.time+=1;invalid.rgb=None;invalid.valid[:]=False;invalid.distances[:]=np.nan
        fresh_memory=Memory(arch,cfg);fresh_memory.fuse(invalid)
        assert all(r.traversability=='unverified' for r in fresh_memory.records.values())
        fresh_memory.close();memory.close()
        sensor_cases.append(dict(variant=variant,opening=o['id'],expected=expected,actual=actual,
                                 duplicate_frame_rejected=True,missing_data_not_free=True))
        s.close()
    report['real_sensor_variants']=sensor_cases
    report['scope']='static geometry + real sensor captures + math + model mapping; complete-task evidence is in run folders'
    save_json(ROOT/'results'/'verification.json',report);print('PASS: candidate math, 16 geometry cases, 3 inertia mappings, 6 real sensor content cases')
    return report

if __name__=='__main__':run()
