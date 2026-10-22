"""1024 full-vehicle MuJoCo rollouts, state-specific, resumable and versioned.

All elapsed physics time counts; failures have +inf cost, never an invented
penalty. Main strategy receives predictions, not true environmental parameters.
"""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import os
import time
import runtime
import numpy as np
import mujoco
from configuration import ROOT
from navigation.physics import World
from navigation.sampling import candidate_worlds
from navigation.geometry import door_comparison
from navigation.interactions import (approach,contact_pose,merge_pose,route,probe,
                                     initialize_prediction,observed_mechanical_state)

_worker=None

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,(np.integer,int)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if hasattr(x,'as_posix'):return str(x)
    return x

def save_json(path,obj):
    path.write_text(json.dumps(safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def init_worker(arch,cfg,opening,state,mode,contact_state,geo):
    global _worker
    w=World(arch,cfg,visual=False,observed_contents={opening['id']:'door'})
    _worker=(w,opening,state,mode,contact_state,geo)

def candidate(item):
    index,theta=item;w,o,state,mode,contact_state,geo=_worker
    panel=initialize_prediction(w,state,o,theta)
    jd=route(w,o,panel,'door',geo)
    if mode=='initial':
        panel=initialize_prediction(w,contact_state,o,theta)
        z,stream=probe(w,o,panel)
        return index,jd,None,z,stream
    panel=initialize_prediction(w,state,o,theta)
    jb=route(w,o,panel,'bypass',geo)
    return index,jd,jb,None,None

def signature(arch,cfg,opening,state,mode):
    conf=asdict(cfg);conf.pop('workers')
    digest=hashlib.sha256(json.dumps(dict(arch=arch,cfg=conf,opening=opening,state=state,mode=mode,
                                        mujoco=mujoco.__version__,schema=4),sort_keys=True).encode())
    for path in [ROOT/'navigation'/n for n in ('physics.py','interactions.py','geometry.py','sampling.py','sensors.py','perception.py')]+[ROOT/'scene'/'build.py',ROOT/'assets'/'models'/'tensegrity_real_model.xml',ROOT/'experiments'/'library.py']:
        digest.update(path.read_bytes())
    for path in sorted((ROOT/'assets'/'textures').glob('*.png')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()

def build(arch,cfg,opening,state,mode='initial',progress=None):
    theta=candidate_worlds(cfg);n=len(theta);key=signature(arch,cfg,opening,state,mode)
    folder=ROOT/'results'/'cache'/(opening['id']+'_'+mode+'_'+key[:16]);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'worlds.npz'
    arrays=dict(theta=theta,done=np.zeros(n,bool),J_door=np.full(n,np.nan),J_bypass=np.full(n,np.nan),
                elapsed_door=np.zeros(n),elapsed_bypass=np.zeros(n),door_reason=np.full(n,'',dtype='U100'),
                bypass_reason=np.full(n,'',dtype='U100'),z_hat=np.full((n,2),np.nan),
                stream=np.full((n,24,2),np.nan),wall_peak=np.zeros(n),contact_peak=np.zeros(n))
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if str(old['signature'])!=key:raise RuntimeError('invalid cache signature')
            for k in arrays:arrays[k]=old[k].copy()
        if not np.array_equal(arrays['theta'],theta) or arrays['done'].shape!=(n,):
            raise RuntimeError('cache candidate domain/shape mismatch')
    def save():
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,signature=np.array(key),**arrays)
        os.replace(tmp,path)
    if arrays['done'].all():
        if np.isnan(arrays['J_door']).any() or np.isnan(arrays['J_bypass']).any():
            raise RuntimeError('complete flag with invalid route costs')
        if mode=='initial' and (not np.isfinite(arrays['z_hat']).all() or not np.isfinite(arrays['stream']).all()):
            raise RuntimeError('invalid completed Probe predictions')
        if any('numerical_anomaly' in r for r in arrays['door_reason']) or any('numerical_anomaly' in r for r in arrays['bypass_reason']):
            raise RuntimeError('numerical-anomaly cache is not valid for decisions')
        print('Candidate cache:',folder.name,flush=True)
        return dict(**arrays,signature=key,folder=folder,labels=(arrays['J_door']<arrays['J_bypass']).astype(int))
    o=opening;startpos=np.array(state['robot'][-1]['position_quaternion'][:3])
    # The frame joint is located by name, not presumed joint order.
    for r in state['robot']:
        if r['joint']=='frame_joint':startpos=np.array(r['position_quaternion'][:3])
    geo=door_comparison(arch,cfg,o,approach(o) if mode!='initial' else startpos,merge_pose(o))
    helper=World(arch,cfg,visual=False,observed_contents={o['id']:'door'})
    panel=initialize_prediction(helper,state,o,theta[0]);contact_state=None
    if mode=='initial':
        helper.fly([contact_pose(o)],yaw=0.,speed=.35)
        if helper.contact_peak>.05:raise RuntimeError('standard approach must be free of contact')
        contact_state=observed_mechanical_state(helper,o,panel)
        panel=initialize_prediction(helper,state,o,theta[0]);bypass=route(helper,o,panel,'bypass',geo)
        if not bypass['success'] or bypass['contact_peak_N']>.05:
            raise RuntimeError('shared bypass failed or contacted environment: '+str(bypass))
        arrays['J_bypass'][:]=bypass['time_s'];arrays['elapsed_bypass'][:]=bypass['elapsed_s']
    pending=[(i,theta[i]) for i in range(n) if not arrays['done'][i]];started=time.perf_counter()
    print('Computing',mode,o['id'],len(pending),'worlds, workers=',cfg.workers,flush=True)
    def accept(row):
        i,jd,jb,z,stream=row
        arrays['J_door'][i]=jd['time_s'];arrays['elapsed_door'][i]=jd['elapsed_s'];arrays['door_reason'][i]=jd['reason']
        arrays['wall_peak'][i]=jd['wall_peak_N'];arrays['contact_peak'][i]=jd['contact_peak_N']
        if jb is not None:
            arrays['J_bypass'][i]=jb['time_s'];arrays['elapsed_bypass'][i]=jb['elapsed_s'];arrays['bypass_reason'][i]=jb['reason']
        if z is not None:arrays['z_hat'][i]=z;arrays['stream'][i]=stream
        arrays['done'][i]=True
        done=int(arrays['done'].sum())
        if done%32==0 or done==n:
            save();print('  {}/{} {:.1f}s'.format(done,n,time.perf_counter()-started),flush=True)
        if progress:progress(done,n)
    try:
        if cfg.workers==1:
            init_worker(arch,cfg,o,state,mode,contact_state,geo)
            for item in pending:accept(candidate(item))
        else:
            with ProcessPoolExecutor(max_workers=cfg.workers,initializer=init_worker,initargs=(arch,cfg,o,state,mode,contact_state,geo)) as pool:
                items=iter(pending)
                # Bound queued work so closing a window cancels unstarted worlds
                # instead of waiting for all 1024 submitted jobs to finish.
                futures=set()
                def submit_one():
                    item=next(items,None)
                    if item is not None:futures.add(pool.submit(candidate,item))
                for _ in range(cfg.workers*2):submit_one()
                try:
                    while futures:
                        done,remaining=wait(futures,timeout=.05,return_when=FIRST_COMPLETED)
                        futures=remaining
                        for future in done:accept(future.result());submit_one()
                        if not done and progress:progress(int(arrays['done'].sum()),n)
                except BaseException:
                    for future in futures:future.cancel()
                    raise
    finally:save()
    if not arrays['done'].all() or np.isnan(arrays['J_door']).any() or np.isnan(arrays['J_bypass']).any():raise RuntimeError('incomplete library')
    if any('numerical_anomaly' in r for r in arrays['door_reason']) or any('numerical_anomaly' in r for r in arrays['bypass_reason']):
        raise RuntimeError('numerical anomalies invalidate this library; see cache')
    labels=(arrays['J_door']<arrays['J_bypass']).astype(int)
    save_json(folder/'metadata.json',dict(signature=key,N=n,mode=mode,seconds=time.perf_counter()-started,
        observed_initial_state=state,geometry=geo,door_success=int(np.isfinite(arrays['J_door']).sum()),door_support=int(labels.sum()),
        both_failed=int((~np.isfinite(arrays['J_door'])&~np.isfinite(arrays['J_bypass'])).sum()),
        assumptions='other anonymous openings optimistically empty in both branches; exact shared public walls/jambs',
        bypass_reuse='one verified zero-contact rollout' if mode=='initial' else 'each candidate recomputed from identical observed post-Probe state'))
    return dict(**arrays,signature=key,folder=folder,labels=labels)
