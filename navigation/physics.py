"""Original flexible vehicle, rotor-only dynamics, virtual proprioception/contact."""
import math
import runtime
import mujoco
import numpy as np
from scene.build import build_xml

def unit(x):return x/max(float(np.linalg.norm(x)),1e-12)
def clip_norm(x,limit):return x*min(1.,limit/max(float(np.linalg.norm(x)),1e-12))

def allocation(mixer,limits,thrust,torque):
    raw=mixer@np.r_[thrust,torque];low,high=limits[:,0],limits[:,1]
    if np.all(raw>=low) and np.all(raw<=high):return raw
    c=mixer[:,0]; thrust=float(np.clip(thrust,np.max(low/c),np.min(high/c)));out=c*thrust
    for diff in (mixer[:,1:3]@torque[:2],mixer[:,3]*torque[2]):
        pos,neg=diff>0,diff<0
        scale=min(1.,float(np.min((high[pos]-out[pos])/diff[pos],initial=1.)),float(np.min((low[neg]-out[neg])/diff[neg],initial=1.)))
        out+=max(0.,scale)*diff
    return out

class PhysicsFailure(RuntimeError):pass

class World:
    def __init__(self,arch,cfg,variant='standard',door_parameters=None,visual=True,observed_contents=None):
        self.arch,self.cfg=arch,cfg
        xml,self.truth=build_xml(arch,cfg,variant,door_parameters,visual,observed_contents)
        self.xml=xml;self.model=m=mujoco.MjModel.from_xml_string(xml);self.data=d=mujoco.MjData(m)
        self.frame_id=m.body('quad_frame').id
        self.robot_ids={m.body(n).id for n in ('link1','link2','link3','link4','quad_frame')}
        for bid in range(1,m.nbody):
            if int(m.body_parentid[bid]) in self.robot_ids:self.robot_ids.add(bid)
        self.robot_geoms={g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.robot_ids}
        self.motor_ids=np.array([m.actuator('thruster'+str(i)).id for i in range(1,5)])
        self.mass=float(m.body_mass[list(self.robot_ids)].sum())
        cols=[]
        for a in self.motor_ids:
            sid=int(m.actuator_trnid[a,0]);f=m.actuator_gear[a,:3];t=np.cross(m.site_pos[sid],f)+m.actuator_gear[a,3:]
            cols.append(np.r_[f[2],t])
        self.mixer=np.linalg.inv(np.column_stack(cols));self.limits=m.actuator_ctrlrange[self.motor_ids].copy()
        self.force=self.wall_force=self.floor_force=0.;self.step_count=0;self.callback=None
        self.stage='INITIAL_SETTLE';self.target=np.array(arch['start_xyz_m']);self.yaw_reference=self.yaw_target=0.
        self.ctrl=np.zeros(4);self.wrench=np.zeros(6);self.contact_peak=0.;self.wall_peak=0.;self.min_height=10.;self.max_height=0.
        self.max_tilt=0.;self.max_command=0.;self.last_horizontal=np.zeros(2);self.saturation_steps=0
        mujoco.mj_forward(m,d);delta=self.target-d.sensor('pos').data
        # Legal one-time rigid initialization. Never called by the live flight loop.
        for jid in range(m.njnt):
            if m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[jid]) in self.robot_ids:
                a=int(m.jnt_qposadr[jid]);d.qpos[a:a+3]+=delta
        mujoco.mj_forward(m,d)

    def observed(self):
        d=self.data
        return (d.sensor('pos').data.copy(),d.sensor('linvel').data.copy(),
                d.xmat[self.frame_id].reshape(3,3).copy(),d.sensor('imu_gyro').data.copy())

    def control(self,horizontal=None,yaw=None,speed=None):
        pos,vel,R,gyro=self.observed();cfg=self.cfg
        recovering=self.stage in ('BACKOFF','RECOVER','PUSH_BRAKE')
        error=self.target-pos;v=clip_norm(1.25*error[:2],cfg.flight_speed if speed is None else speed)
        acc=np.r_[3.2*(v-vel[:2]),np.clip(10*error[2]-5*vel[2],-4,4)]
        acc[:2]=clip_norm(acc[:2],4. if recovering else 2.)
        force=self.mass*(acc+[0,0,9.81])
        if horizontal is not None:force[:2]=horizontal
        self.last_horizontal=force[:2].copy()
        if yaw is not None:self.yaw_target=float(yaw)
        dyaw=(self.yaw_target-self.yaw_reference+math.pi)%(2*math.pi)-math.pi
        self.yaw_reference+=float(np.clip(dyaw,-.007,.007))
        b3=unit(force);b2=unit(np.cross(b3,[math.cos(self.yaw_reference),math.sin(self.yaw_reference),0]))
        desired=np.column_stack((np.cross(b2,b3),b2,b3));e=.5*(desired.T@R-R.T@desired)
        er=np.array([e[2,1],e[0,2],e[1,0]]);authority=1.6 if recovering else 1.
        torque=-np.array([.65*authority,.65*authority,.35])*er-np.array([.1,.1,.08])*gyro
        thrust=max(0.,float(force@R[:,2]))
        if recovering:thrust=max(0.,float(force[2])/max(float(R[2,2]),.35))
        return allocation(self.mixer,self.limits,thrust,torque)

    def step(self,horizontal=None,yaw=None,speed=None):
        m,d=self.model,self.data;mujoco.mj_step1(m,d)
        if self.step_count%5==0:self.ctrl=self.control(horizontal,yaw,speed)
        d.ctrl[:]=0.;d.ctrl[self.motor_ids]=self.ctrl;mujoco.mj_step2(m,d)
        self.force=self.wall_force=self.floor_force=0.
        # Virtual frame contact instrument: sum ALL external robot contacts,
        # never select the target by environment geom ID for online feedback.
        for ci in range(d.ncon):
            con=d.contact[ci];a,b=int(con.geom1),int(con.geom2)
            if (a in self.robot_geoms)==(b in self.robot_geoms):continue
            other=b if a in self.robot_geoms else a
            mujoco.mj_contactForce(m,d,ci,self.wrench);normal=max(0.,float(self.wrench[0]));self.force+=normal
            # These two channels are evaluator diagnostics only.
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
            if name.startswith('wall_') or name.startswith('jamb_'):self.wall_force+=normal
            if name=='ground':self.floor_force+=normal
        self.contact_peak=max(self.contact_peak,self.force);self.wall_peak=max(self.wall_peak,self.wall_force)
        self.step_count+=1
        if self.step_count%25==0:
            mujoco.mj_forward(m,d);pos,vel,R,gyro=self.observed()
            self.min_height=min(self.min_height,float(pos[2]));self.max_height=max(self.max_height,float(pos[2]))
            self.max_tilt=max(self.max_tilt,math.degrees(math.acos(float(np.clip(R[2,2],-1,1)))))
            if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise PhysicsFailure('numerical_anomaly')
            if pos[2]<.50 or pos[2]>2.05:raise PhysicsFailure('flight_height_violation')
            if not (-5.12<pos[0]<5.12 and .35<pos[1]<12.65):raise PhysicsFailure('boundary_clearance_violation')
            if self.callback:self.callback(self)

    def hold(self,target,duration,stage='OBSERVE',yaw=None):
        self.target=np.r_[np.asarray(target)[:2],1.25];self.stage=stage;end=self.data.time+duration
        while self.data.time<end-1e-9:self.step(yaw=yaw)

    def fly(self,path,settle=True,stage='FLIGHT',yaw=None,speed=None):
        self.stage=stage
        for point in path:
            self.target=np.r_[np.asarray(point)[:2],1.25];start=self.data.time;dwell=0.
            delta=self.target-self.observed()[0]
            heading=math.atan2(-delta[0],delta[1]) if np.linalg.norm(delta[:2])>.2 else self.yaw_target
            if yaw is not None:heading=yaw
            while self.data.time-start<45.:
                self.step(yaw=heading,speed=speed)
                if self.step_count%5:continue
                p,v,R,g=self.observed();reached=np.linalg.norm(p-self.target)<(.065 if settle else .085)
                if settle:
                    angle=math.atan2(R[1,0],R[0,0]);err=(angle-heading+math.pi)%(2*math.pi)-math.pi
                    reached &= np.linalg.norm(v)<.065 and np.linalg.norm(g)<.16 and abs(err)<.11
                    dwell=dwell+.01 if reached else 0.;reached=dwell>=.25
                if reached:break
            else:raise PhysicsFailure('waypoint_timeout '+str(self.target.tolist()))
        return True

    def snapshot(self):
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),act=self.data.act.copy(),
                    time=float(self.data.time),yaw_reference=self.yaw_reference)

    def restore_clone(self,state):
        """Prediction clones only. The online mission never calls this method."""
        mujoco.mj_resetData(self.model,self.data)
        for key in ('qpos','qvel','act'):getattr(self.data,key)[:]=state[key]
        self.data.time=state['time'];self.yaw_reference=self.yaw_target=state['yaw_reference']
        mujoco.mj_forward(self.model,self.data);self.step_count=0
