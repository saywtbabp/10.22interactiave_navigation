"""Continuous perception -> memory -> planning -> physical action -> recovery."""
from dataclasses import asdict
from datetime import datetime
import csv
import math
import time
import traceback
import runtime
import numpy as np
from PIL import Image
from configuration import ROOT,architecture
from .physics import World,PhysicsFailure,clip_norm
from .sensors import Sensors
from .perception import Memory
from .geometry import Planner,door_comparison
from .contact_geometry import contact_comparison
from .interactions import (approach,contact_pose,merge_pose,probe,push,recover,panel_geometry,
                           observed_mechanical_state)
from .belief import decide,support,consistency_update
from .sampling import candidate_worlds
from experiments.library import build,save_json,safe

class Mission:
    def __init__(self,cfg,variant='standard',door_parameters=None,baseline='adaptive',output=None,ui=None):
        self.cfg=cfg;self.arch=architecture();self.world=World(self.arch,cfg,variant,door_parameters)
        self.sensors=Sensors(self.world,cfg);self.memory=Memory(self.arch,cfg);self.ui=ui
        self.variant,self.baseline=variant,baseline
        self.folder=output or ROOT/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+variant+'_'+baseline)
        self.folder.mkdir(parents=True,exist_ok=True)
        # Evaluator output only. No policy method reads this file or mapping.
        save_json(self.folder/'scene_truth_for_evaluation.json',self.world.truth)
        self.status='初始化';self.reason='固定高度悬停与传感器启动';self.current=None;self.action='Observe'
        self.path=np.empty((0,2));self.trajectory=[];self.telemetry=[];self.sensor_log=[];self.decisions=[];self.weight_history=[]
        self.replans=0;self.probes=0;self.success=False;self.failure='';self.prediction=False;self.progress=(0,1024)
        self.door_beliefs={};self.evaluation=[]
        self.inertial=[]
        self.belief_owner=None
        self.weights=np.ones(1<<cfg.sobol_power)/(1<<cfg.sobol_power);self.labels=None;self.theta=candidate_worlds(cfg)
        self.metrics=dict(G=None,p_D=None,D=None);self.force_peak=0.;self.saved_keys=set();self.wall_started=time.perf_counter()
        self._log=(self.folder/'sensors.jsonl').open('w',encoding='utf-8');self._last_image=-10.
        self.world.callback=self.on_step
        if ui:ui.attach(self)

    def on_step(self,w):
        p,v,R,g=w.observed();self.trajectory.append([float(w.data.time),*p.tolist()])
        self.inertial.append(dict(time_s=float(w.data.time),imu_acc_m_s2=w.data.sensor('imu_acc').data.copy(),
                                  gyro_body_rad_s=g.copy(),source='original onboard IMU sensors; logged at 20 Hz'))
        self.telemetry.append([float(w.data.time),w.stage,*p.tolist(),*v.tolist(),*g.tolist(),*w.ctrl.tolist(),w.force,w.wall_force,
                               math.degrees(math.acos(float(np.clip(R[2,2],-1,1))))])
        packet,fresh=self.sensors.acquire()
        if fresh:
            self.memory.fuse(packet)
            record=dict(time_s=packet.time,position=packet.position,rotation=packet.rotation,
                        camera_position=packet.camera_position,optical_rotation=packet.optical_rotation,
                        focal_px=packet.focal,detections=self.memory.detections,
                        distances_m=packet.distances,valid=packet.valid,returns=packet.returns)
            self._log.write(__import__('json').dumps(safe(record),ensure_ascii=False,allow_nan=False)+'\n')
            # Time-stamped evidence including frames on every interaction, not a fake replay.
            if packet.rgb is not None and (packet.time-self._last_image>=2. or w.stage in ('PROBE','GLASS_VERIFY','PUSH','BACKOFF') and packet.time-self._last_image>=.4):
                frames=self.folder/'camera';frames.mkdir(exist_ok=True)
                Image.fromarray(packet.rgb).save(frames/('{:09.3f}_{}.jpg'.format(packet.time,w.stage)),quality=85)
                self._last_image=packet.time
        if self.ui:self.ui.tick()
        from experiments.truth import sample
        self.evaluation.append(sample(w,self.memory))

    def state_panel(self,rec):
        # A stale fit is retained as memory but cannot silently authorize crossing.
        return float(rec.angle),float(rec.angular_velocity)

    def planner(self,extra_sealed=()):
        sealed=list(extra_sealed);panels=[]
        for oid,rec in self.memory.records.items():
            if rec.traversability=='blocked' or rec.bypass_selected:sealed.append(oid)
            elif rec.semantic=='door' and rec.traversability=='free':panels.append(panel_geometry(rec.geometry,rec.angle)[2])
        return Planner(self.arch,self.cfg,panels=panels,sealed=sealed)

    def plan(self,goal):
        path,_=self.planner().plan(self.world.observed()[0],goal);self.path=path;self.replans+=1
        self.memory.event(self.world.data.time,self.current or '', 'replan',count=self.replans)
        return path

    def next_opening(self,path):
        position=self.world.observed()[0]
        for a,b in zip(path[:-1],path[1:]):
            if abs(b[1]-a[1])<1e-8:continue
            candidates=[]
            for oid,rec in self.memory.records.items():
                x,y=rec.geometry['center_xy_m']
                if y<position[1]+.25:continue
                f=(y-a[1])/(b[1]-a[1])
                if 0<=f<=1:
                    xx=a[0]+f*(b[0]-a[0])
                    if abs(xx-x)<rec.geometry['width_m']/2:candidates.append((f,oid))
            if candidates:return min(candidates)[1]
        return None

    def save_key(self,label):
        if label in self.saved_keys:return
        self.saved_keys.add(label)
        p=self.sensors.packet
        if p is not None and p.rgb is not None:Image.fromarray(p.rgb).save(self.folder/(label+'_rgb.png'))
        if self.ui:self.ui.save(self.folder/label)
        else:
            from ui.art import save_views
            save_views(self,self.folder/label)

    def prediction_progress(self,done,total):
        self.progress=(done,total)
        if self.ui:self.ui.tick(predicting=True)

    def door(self,rec):
        w=self.world;o=rec.geometry;oid=rec.stable_id;self.action='Evaluate';self.status='门路线比较'
        geo=door_comparison(self.arch,self.cfg,o,w.observed()[0],merge_pose(o));G=geo['G']
        self.metrics=dict(G=G,p_D=None,D=None);self.weights=np.ones(len(self.theta))/len(self.theta);self.labels=None
        self.belief_owner=oid
        self.door_beliefs[oid]=dict(theta=self.theta.copy(),weights=self.weights.copy(),labels=None,G=G,
                                   p_D=None,D=None,probe_used=False,source='independent uniform prior')
        panel=lambda:self.state_panel(rec)
        if self.baseline=='always_avoid' or G<self.cfg.geometry_threshold and self.baseline!='always_push':
            self.action='Avoid';self.reason='G < 3 m，几何收益不足' if self.baseline!='always_avoid' else '始终绕行基线'
            self.decisions.append(dict(time_s=w.data.time,opening=oid,**self.metrics,action=self.action,reason=self.reason,
                                       N=len(self.theta),labels_source='not evaluated: geometry gate short-circuited',geometry=geo))
            rec.bypass_selected=True;self.memory.event(w.data.time,oid,'avoid_selected',G=G);return
        if self.baseline=='always_push':
            self.action='Push';self.reason='始终尝试推门基线；超时后安全恢复绕行'
            self.decisions.append(dict(time_s=w.data.time,opening=oid,**self.metrics,action=self.action,reason=self.reason,
                                       N=len(self.theta),labels_source='fixed-action explicit baseline, no inference',geometry=geo))
            self.execute_push(rec)
            return
        self.prediction=True;self.status='预测中 / 仿真暂停';self.reason='计算 1024 个候选世界的实际路线'
        state=observed_mechanical_state(w,o,panel)
        library=build(self.arch,self.cfg,o,state,progress=self.prediction_progress)
        self.prediction=False;self.labels=library['labels'];p,D=support(self.weights,self.labels)
        action,reason=decide(G,p,D,self.cfg)
        if self.baseline=='always_push':action,reason='Push','always-push explicit baseline'
        if self.baseline=='no_probe' and action=='Probe':action,reason=('Push' if p>.5 else 'Avoid'),'no-Probe majority baseline'
        self.metrics=dict(G=G,p_D=p,D=D);self.action=action;self.reason=reason
        self.door_beliefs[oid].update(weights=self.weights.copy(),labels=self.labels.copy(),p_D=p,D=D,
                                     source=str(library['folder']))
        event=dict(time_s=w.data.time,opening=oid,**self.metrics,action=action,reason=reason,N=len(self.theta),geometry=geo,
                   labels_source=str(library['folder']),prior='uniform 1/N',probe_used=False)
        self.decisions.append(event);print('DOOR',oid,'G',G,'p',p,'D',D,'->',action,flush=True)
        self.save_key(oid+'_decision')
        if action=='Probe':
            self.status='标准主动探测';rec.probe_used=True;self.probes+=1
            w.fly([contact_pose(o)],yaw=0.,speed=.35)
            # Align exposure and interaction clock before comparing prefixes.
            for _ in range((25-w.step_count%25)%25):w.step(yaw=0.)
            packet,_=self.sensors.acquire(force=True);self.memory.fuse(packet)
            prior=self.weights.copy();probe_start=w.data.time
            def update(t,z):
                if abs(t/.1-round(t/.1))>.01:return
                index=min(23,max(0,int(round(t/.05))-1))
                self.weights,_=consistency_update(prior,library['stream'][:,index,:],z,self.cfg)
                pp,dd=support(self.weights,self.labels);self.metrics.update(p_D=pp,D=dd)
                self.weight_history.append(dict(opening=oid,time_s=w.data.time,probe_elapsed_s=t,weights=self.weights.copy(),observation=z.copy(),p_D=pp,D=dd))
            z,stream=probe(w,o,panel,progress=update)
            self.weights,error=consistency_update(prior,library['z_hat'],z,self.cfg)
            state=observed_mechanical_state(w,o,panel)
            self.prediction=True;self.status='探测后预测 / 仿真暂停';self.reason='从同一观测后状态重算两条剩余路线'
            post=build(self.arch,self.cfg,o,state,mode='post_probe',progress=self.prediction_progress)
            self.prediction=False;self.labels=post['labels'];p,D=support(self.weights,self.labels)
            # Same M and actual post-Probe start; target door branch forced, alternative sealed.
            postgeo=contact_comparison(self.arch,self.cfg,o,w.observed()[0],merge_pose(o))
            Gpost=postgeo['G'];action,reason=decide(Gpost,p,D,self.cfg,probe_used=True)
            self.metrics=dict(G=Gpost,p_D=p,D=D);self.action=action;self.reason=reason
            failed_both=~np.isfinite(post['J_door'])&~np.isfinite(post['J_bypass'])
            if float(self.weights[failed_both].sum())>.1:
                action='Avoid';reason='both-route control failures retain material posterior mass; unresolved conservative fallback'
                self.action=action;self.reason=reason
            self.door_beliefs[oid].update(weights=self.weights.copy(),labels=self.labels.copy(),G=Gpost,p_D=p,D=D,
                                         probe_used=True,source=str(post['folder']))
            self.decisions.append(dict(time_s=w.data.time,opening=oid,**self.metrics,action=action,reason=reason,
                 N=len(self.theta),probe_used=True,observation=z,probe_duration_s=float(w.data.time-probe_start),
                 labels_source=str(post['folder']),same_post_state=state,geometry=postgeo,
                 both_failed_worlds=int((~np.isfinite(post['J_door'])&~np.isfinite(post['J_bypass'])).sum()),
                 both_failed_weight=float(self.weights[~np.isfinite(post['J_door'])&~np.isfinite(post['J_bypass'])].sum()),
                 posterior_source='one likelihood from original prior; no repeated-prefix multiplication'))
            print('POST',oid,'p',p,'D',D,'->',action,flush=True);self.save_key(oid+'_post_probe')
        if action=='Avoid':
            if rec.probe_used:
                self.status='脱离与恢复';path=recover(w,o,panel,bypass=True);w.fly(path[1:],yaw=0.)
            rec.bypass_selected=True;self.memory.event(w.data.time,oid,'avoid_selected',probe_used=rec.probe_used);return
        self.execute_push(rec)

    def execute_push(self,rec):
        w=self.world;o=rec.geometry;oid=rec.stable_id;panel=lambda:self.state_panel(rec)
        self.status='推门执行';self.reason='保留 Probe 后验，正式 Push 不复用标准 Probe 观测库'
        if w.observed()[0][1]<o['center_xy_m'][1]-.6:w.fly([contact_pose(o)],yaw=0.,speed=.35)
        if not push(w,o,panel):
            self.reason='实际 Push 超时，恢复并绕行';path=recover(w,o,panel,bypass=True);w.fly(path[1:],yaw=0.)
            rec.bypass_selected=True;self.memory.event(w.data.time,oid,'push_failed_then_avoid');return
        self.status='脱离与恢复';path=recover(w,o,panel)
        if w.data.time-rec.angle_time>.25:raise PhysicsFailure('stale_panel_geometry_before_crossing')
        rec.traversability='free';rec.confirmation='confirmed';self.memory.event(w.data.time,oid,'door_geometry_clear',estimated_angle_rad=rec.angle)
        self.path=path;w.fly(path[1:],yaw=0.);self.save_key(oid+'_crossed')

    def glass(self,rec):
        w=self.world;o=rec.geometry;oid=rec.stable_id;x,y=o['center_xy_m']
        self.status='歧义开口接触验证';self.action='Verify';self.reason='疑似玻璃独立低速验证，不进入门 G/D 门控'
        self.metrics=dict(G=None,p_D=None,D=None)
        # Slowly approach the expected surface using only rotor feedback.
        w.fly([[x+.12,y-.47]],yaw=0.,speed=.20)
        w.target=np.array([x+.12,y+.10,1.25]);w.stage='GLASS_VERIFY';start=w.data.time;peak=0.;blocked=False;valid_free=False
        while w.data.time-start<5:
            w.step(yaw=0.,speed=.12);peak=max(peak,w.force)
            p=w.observed()[0]
            if w.force>.35:blocked=True;break
            if p[1]>=y+.08:
                valid_free=rec.range_evidence.get('clearance_support',0)>.8 and w.data.time-rec.last_seen_time<.3
                break
        if blocked:
            actual_position,actual_velocity,actual_rotation,_=w.observed()
            self.memory.verified(oid,float(w.data.time),'blocked',peak,semantic_preserved=rec.semantic,
                                 observed_position=actual_position.tolist(),observed_speed_m_s=float(np.linalg.norm(actual_velocity)),
                                 approach_speed_limit_m_s=.12)
            self.status='玻璃受阻 / 退离恢复';w.stage='BACKOFF';w.target=np.array([x+.12,y-.85,1.25])
            stable=0.;start=w.data.time
            while w.data.time-start<10:
                w.step(yaw=0.,speed=.30)
                p,v,R,g=w.observed();ok=w.force<.1 and np.linalg.norm(v)<.12 and R[2,2]>.96 and np.linalg.norm(g)<.5 and p[1]<y-.70
                stable=stable+self.cfg.timestep if ok else 0.
                if stable>.2:break
            else:raise PhysicsFailure('glass_recovery_timeout')
            self.save_key(oid+'_glass_blocked')
        elif valid_free:
            self.memory.verified(oid,float(w.data.time),'free',peak,valid_validation_position=True)
            w.fly([[x,y+.75]],yaw=0.)
        else:
            rec.bypass_selected=True;self.memory.event(w.data.time,oid,'verification_inconclusive')
            w.fly([[x,y-.85]],yaw=0.,stage='BACKOFF')

    def run(self):
        try:
            w=self.world;w.hold(self.arch['start_xyz_m'],.8,stage='INITIAL_SETTLE',yaw=0.)
            self.save_key('00_initial')
            for _ in range(24):
                if np.linalg.norm(w.observed()[0]-self.arch['goal_xyz_m'])<.10:self.success=True;break
                self.status='自主导航';self.reason='依据当前认知地图搜索可行路径';path=self.plan(self.arch['goal_xyz_m'])
                oid=self.next_opening(path)
                if oid is None:w.fly(path[1:]);continue
                self.current=oid;rec=self.memory.records[oid];o=rec.geometry;x,y=o['center_xy_m']
                if rec.traversability=='free':
                    dest=[x,y+.72];self.status='净空通行';self.action='Cross';self.reason='有效多帧测距已确认净空'
                    p=self.plan(dest);w.fly(p[1:]);self.save_key(oid+'_free_crossed');continue
                dest=approach(o);p=self.plan(dest);w.fly(p[1:],yaw=0.)
                self.status='观察匿名开口';self.action='Observe';self.reason='同步 RGB / 测距，多帧确认'
                w.hold(dest,1.1,yaw=0.)
                if rec.semantic=='door':self.door(rec)
                elif rec.semantic=='suspected_glass':self.glass(rec)
                elif rec.traversability=='free':
                    self.status='净空通行';p=self.plan([x,y+.72]);w.fly(p[1:],yaw=0.);self.save_key(oid+'_free_crossed')
                else:
                    rec.bypass_selected=True;self.memory.event(w.data.time,oid,'insufficient_evidence_avoid')
            if not self.success and np.linalg.norm(w.observed()[0]-self.arch['goal_xyz_m'])<.10:self.success=True
            if not self.success:raise PhysicsFailure('navigation_iteration_limit')
            self.status='任务完成';self.action='Hover';self.reason='已到达目标并稳定悬停'
            w.hold(self.arch['goal_xyz_m'],.6,stage='COMPLETE',yaw=0.)
            self.save_key('99_complete')
        except Exception as exc:
            self.failure=str(exc);self.status='任务未完成';self.reason=str(exc)
            (self.folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            print(traceback.format_exc(),flush=True)
        finally:self.finish()
        return self.success

    def finish(self):
        self._log.close();w=self.world
        with (self.folder/'flight.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['time_s','state','x','y','z','vx','vy','vz','gx','gy','gz','motor1_N','motor2_N','motor3_N','motor4_N','external_contact_N','eval_wall_contact_N','tilt_deg']);writer.writerows(self.telemetry)
        save_json(self.folder/'online_map.json',self.memory.snapshot());save_json(self.folder/'events.json',self.memory.events)
        save_json(self.folder/'decisions.json',self.decisions);save_json(self.folder/'weight_history.json',self.weight_history)
        save_json(self.folder/'door_beliefs.json',self.door_beliefs);save_json(self.folder/'evaluation_truth.json',self.evaluation)
        save_json(self.folder/'imu_measurements.json',self.inertial)
        self.summary=dict(success=self.success,failure=self.failure,simulated_time_s=float(w.data.time),
            wall_time_s=time.perf_counter()-self.wall_started,config=asdict(self.cfg),variant=self.variant,baseline=self.baseline,
            probes=self.probes,replans=self.replans,map_version=self.memory.version,final_position=w.observed()[0],
            contact_peak_N=w.contact_peak,wall_contact_peak_N=w.wall_peak,height_range_m=[w.min_height,w.max_height],max_tilt_deg=w.max_tilt,
            sensor_frames=self.sensors.frames,decision_count=len(self.decisions),
            assumptions=['ideal localization/AHRS and own multibody proprioception','RGB/range engineering detector, not YOLO',
                         'virtual all-external-contact normal force instrument','lidar panel angle fit, not hinge encoder'])
        save_json(self.folder/'summary.json',self.summary)
        from experiments.report import generate
        generate(self)
        print('RESULT',safe(self.summary),flush=True);print('Saved:',self.folder,flush=True)
