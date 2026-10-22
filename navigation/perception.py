"""Replaceable pixel/range engineering detector and persistent anonymous memory.

No simulator, geom IDs, materials, joint state or hidden contents enter here.
Scores are pixel fractions, NOT trained or calibrated probabilities.
"""
from dataclasses import dataclass,field,asdict
import math
import numpy as np
from .sensors import project

@dataclass
class Record:
    stable_id: str
    geometry: dict
    semantic: str = 'unknown'
    traversability: str = 'unverified'
    confirmation: str = 'candidate'
    visual_evidence: dict = field(default_factory=dict)
    range_evidence: dict = field(default_factory=dict)
    contact_history: list = field(default_factory=list)
    last_seen_time: float = -1.
    last_update_time: float = -1.
    evidence_kind: str = ''
    evidence_count: int = 0
    contradict_count: int = 0
    angle: float = 0.
    angle_time: float = -1.
    angular_velocity: float = 0.
    probe_used: bool = False
    bypass_selected: bool = False

class Memory:
    def __init__(self,arch,cfg):
        self.arch,self.cfg=arch,cfg;self.records={o['id']:Record(o['id'],dict(o)) for o in arch['openings']}
        self.version=0;self.events=[];self.last_frame=-1.;self.detections=[];self.packet=None

    def event(self,time,oid,kind,**info):
        self.version+=1;self.events.append(dict(time_s=float(time),stable_id=oid,event=kind,map_version=self.version,**info))

    def fuse(self,p):
        if p is None or p.time<=self.last_frame:return False
        self.last_frame=p.time;self.packet=p;self.detections=[]
        for oid,rec in self.records.items():
            o=rec.geometry;x,y=o['center_xy_m'];width=o['width_m']
            if rec.semantic=='door':self.track_panel(rec,p)
            delta=np.array([x,y,1.25])-p.camera_position;distance=float(np.linalg.norm(delta[:2]))
            # Main navigation proceeds south-to-north; observations from either side allowed.
            points=[[x-width*.32,y,z] for z in (.65,1.8)]+[[x+width*.32,y,z] for z in (.65,1.8)]
            uv,depth=project(p,points)
            if distance>4.0 or np.any(depth<.30) or p.rgb is None:continue
            h,w=p.rgb.shape[:2];raw=[uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()]
            if raw[0]<-15 or raw[2]>w+15:continue
            x1,y1,x2,y2=[int(v) for v in (max(0,raw[0]),max(0,raw[1]),min(w,raw[2]),min(h,raw[3]))]
            if x2-x1<12 or y2-y1<18:continue
            # Synchronized rays intersecting the central aperture, with plane visibility.
            dy=p.directions[:,1];t=(y-p.position[1])/np.where(abs(dy)>.01,dy,np.nan)
            at=p.position[None,:]+t[:,None]*p.directions
            selected=(t>.05)&(t<self.cfg.scan_range)&(abs(at[:,0]-x)<width*.30)
            idx=np.flatnonzero(selected)
            if len(idx)<3:continue
            usable=idx[p.valid[idx]];ratio=len(usable)/len(idx)
            if len(usable)<3 or ratio<.5:continue
            near=np.abs(p.distances[usable]-t[usable])<.10
            ahead=p.distances[usable]-t[usable]>.30
            behind=p.distances[usable]<t[usable]-.18
            if behind.mean()>.3:continue # Opening occluded: no label from projected ROI.
            rgb=p.rgb[y1:y2,x1:x2].astype(float)/255.;r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
            warm=(r-g>.055)&(g-b>.022)&(r>.10)&(r<.85)
            cool=(b-r>.025)&(g-r>.015)&(b>.15)
            wood=float(warm.mean());glass=float(cool.mean());panel=float(near.mean());clear=float(ahead.mean())
            # Visual texture/chroma plus independent depth. Any weak/invalid ray
            # alone cannot declare glass or empty. Glass appearance remains only suspected.
            if wood>.42 and panel>.38:kind='door';score=wood
            elif glass>.23 and panel>.05:kind='suspected_glass';score=glass
            elif clear>.92 and panel<.03 and wood<.12:kind='free';score=clear
            else:kind='unknown';score=max(wood,glass)
            if p.time-rec.last_seen_time>.5:
                rec.evidence_kind='';rec.evidence_count=0
            rec.last_seen_time=p.time
            rec.visual_evidence=dict(class_name=kind,score=score,bbox_xyxy=[x1,y1,x2,y2],time_s=p.time,
                                     source='RGB aperture texture/chroma + synchronized range (engineering)',wood_fraction=wood,cool_fraction=glass)
            rec.range_evidence=dict(valid_ratio=ratio,panel_support=panel,clearance_support=clear,ray_count=len(idx),time_s=p.time)
            self.detections.append(dict(stable_id=oid,**rec.visual_evidence))
            if kind=='unknown':continue
            if rec.evidence_kind==kind:rec.evidence_count+=1
            else:rec.evidence_kind=kind;rec.evidence_count=1
            if rec.evidence_count<self.cfg.confirmation_frames:continue
            if rec.contact_history or rec.traversability=='free' or rec.bypass_selected:
                contradiction=(kind=='free' and rec.traversability=='blocked') or (kind in ('door','suspected_glass') and rec.traversability=='free')
                if contradiction:
                    rec.contradict_count+=1
                    if rec.contradict_count==3:
                        rec.confirmation='pending_verification'
                        if rec.traversability=='free':rec.traversability='unverified';rec.semantic=kind
                        self.event(p.time,oid,'contradiction_requires_reverification')
                else:rec.contradict_count=0
                continue
            old=(rec.semantic,rec.traversability,rec.confirmation)
            if kind=='free':rec.traversability='free';rec.confirmation='confirmed'
            else:rec.semantic=kind;rec.confirmation='confirmed' if kind=='door' else 'pending_verification'
            rec.last_update_time=p.time
            if old!=(rec.semantic,rec.traversability,rec.confirmation):self.event(p.time,oid,'multi_frame_confirmed',semantic=rec.semantic,traversability=rec.traversability)
        return True

    def track_panel(self,rec,p):
        """Planar lidar fit against inferred western panel pivot; no joint sensors."""
        o=rec.geometry;x,y=o['center_xy_m'];width=o['width_m']-.18
        hinge=np.array([x-width/2,y+.035]);points=p.points[p.valid,:2]
        q=points-hinge
        radius=np.linalg.norm(q,axis=1)
        q=q[(radius>.13)&(radius<width-.06)&(q[:,0]>-.08)&(q[:,1]>-.07)]
        if len(q)<3:return
        angles=np.linspace(0,math.radians(88),177);normals=np.column_stack((-np.sin(angles),np.cos(angles)))
        residual=abs(q@normals.T+.035);counts=(residual<.014).sum(axis=0)
        k=int(np.argmax(counts));mask=residual[:,k]<.018
        if mask.sum()<3:return
        pts=q[mask]
        if np.linalg.norm(np.ptp(pts,axis=0))<.11:return
        centered=pts-pts.mean(axis=0);cov=centered.T@centered;ev,vec=np.linalg.eigh(cov);v=vec[:,-1]
        if v[0]<0:v=-v
        angle=float(np.clip(math.atan2(v[1],v[0]),0,math.radians(88)))
        if rec.angle_time>=0 and p.time>rec.angle_time:
            rec.angular_velocity=(angle-rec.angle)/(p.time-rec.angle_time)
        rec.angle=angle;rec.angle_time=p.time

    def verified(self,oid,time,outcome,force,**kw):
        r=self.records[oid];r.traversability=outcome;r.confirmation='confirmed';r.last_update_time=time
        r.contact_history.append(dict(time_s=time,outcome=outcome,force_peak_N=force,force_source='all_external_robot_contacts',**kw))
        self.event(time,oid,'contact_verification',outcome=outcome,force_peak_N=force,**kw)

    def snapshot(self):return dict(version=self.version,records={k:asdict(v) for k,v in self.records.items()})
