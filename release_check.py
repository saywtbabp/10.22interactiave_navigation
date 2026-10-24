"""Audit current YOLO upgrade evidence; old color runs cannot satisfy this gate."""
import runtime
import hashlib
import json
from pathlib import Path
import numpy as np
from configuration import ROOT, load_config, architecture
from experiments.audit import audit
from experiments.library import signature, save_json


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    cfg=load_config();arch=architecture();out=ROOT/'results'
    model=ROOT/cfg.yolo_model;sha=hashlib.sha256(model.read_bytes()).hexdigest()
    assert read(model.with_name('metadata.json'))['onnx_sha256']==sha
    available=[]
    for folder in sorted(out.glob('20*')):
        if not (folder/'summary.json').exists():continue
        s=read(folder/'summary.json')
        if s.get('vision',{}).get('model_sha256')==sha and s['config']['sobol_power']==10:
            available.append((folder,s,read(folder/'scene_truth_for_evaluation.json')))
    selected={}
    for label,I in [('standard',.071),('resistant',.673)]:
        choices=[x for x in available if x[1]['variant']=='standard' and x[1]['baseline']=='adaptive'
                 and abs(x[2]['O01']['inertia']-I)<1e-9]
        assert choices,'Missing current formal run: '+label
        selected[label]=choices[-1]
    checks={};runs={};libraries={}
    for name,(folder,s,truth) in selected.items():
        a=audit(folder);decisions=read(folder/'decisions.json');beliefs=read(folder/'door_beliefs.json')
        assert s['success'] and a['motor_limits_pass'] and a['fixed_height_pass'] and a['no_wall_contact'],s
        assert [d['action'] for d in decisions]==(['Probe','Push','Avoid'] if name=='standard' else ['Probe','Avoid','Avoid']),decisions
        assert len(beliefs['O01']['weights'])==1024 and np.allclose(beliefs['O05']['weights'],1/1024)
        w=np.array(beliefs['O01']['weights']);labels=np.array(beliefs['O01']['labels'])
        assert abs(w.sum()-1)<1e-10 and abs(float(w@labels)-decisions[1]['p_D'])<1e-10
        verification=read(folder/'verification_trace.json')
        blocked=[v for v in verification if v['outcome']=='blocked']
        if name=='standard':assert any(v['opening']=='O03' for v in blocked)
        for v in verification:
            assert v['outcome'] in ('free','blocked','inconclusive')
            assert 'IMU' in v['method']
            if v['outcome']=='free':assert v['valid_force_samples']>=15
        for d in decisions:
            source=d.get('labels_source','')
            library=Path(source)
            if not library.is_dir():continue
            meta=read(library/'metadata.json');oid=d['opening']
            opening=next(o for o in arch['openings'] if o['id']==oid)
            key=signature(arch,cfg,opening,meta['observed_initial_state'],meta['mode'])
            assert key==meta['signature'] and meta['N']==1024
            with np.load(library/'worlds.npz',allow_pickle=False) as arrays:
                assert arrays['done'].all() and len(arrays['theta'])==1024
                assert not np.isnan(arrays['J_door']).any() and not np.isnan(arrays['J_bypass']).any()
                assert str(arrays['signature'])==key
            libraries[library.name]=dict(N=1024,signature=key,complete=True)
        runs[name]=dict(folder=folder.relative_to(ROOT).as_posix(),summary=s,audit=a,
                        actions=[d['action'] for d in decisions],verifications=[dict(opening=v['opening'],outcome=v['outcome']) for v in verification])
    d0=read(selected['standard'][0]/'decisions.json')[0];d1=read(selected['resistant'][0]/'decisions.json')[0]
    checks['same_unobserved_door_prior']=all(d0[k]==d1[k] for k in ('action','G','p_D','D'))
    unit=read(out/'sensor_upgrade_validation'/'checks.json')
    checks['sensor_regressions']=len(unit)>=12 and all(c['passed'] for c in unit)
    onnx=read(out/'sensor_upgrade_validation'/'onnx_test.json')
    checks['held_out_onnx']=onnx['images']==144 and onnx['model_sha256']==sha
    checks['runtime_cache_memory']=all(v for k,v in read(out/'runtime_verification'/'checks.json').items() if k!='cancel_seconds')
    gui=read(out/'gui_yolo_consistency_all_empty'/'checks.json');checks['real_gui']=all(gui.values())
    gui_standard=read(out/'gui_yolo_consistency_standard'/'checks.json');checks['real_gui_standard']=all(gui_standard.values())
    checks['original_model_unchanged']=(ROOT/'assets/models/tensegrity_real_model.xml').read_bytes()==(ROOT.parent/'Mujoco_0/tensegrity_real_model.xml').read_bytes()
    assert all(checks.values()),checks
    source={}
    paths=list(ROOT.glob('*.py'))
    for directory in ('navigation','scene','ui','experiments','tools'):paths.extend((ROOT/directory).glob('*.py'))
    for path in paths:
        compile(path.read_bytes(),str(path),'exec')
        source[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(checks=checks,runs=runs,current_libraries=libraries,model_sha256=sha,held_out_onnx=onnx,source_sha256=source)
    save_json(out/'yolo_upgrade_release.json',report)
    print('PASS: current trained YOLO, full 1024 standard/resistant missions, IMU verification, real GUI and source integrity',flush=True)


if __name__=='__main__':main()
