"""Evaluator-only diagnostics. Returned values are written to disk, never policy inputs."""
import math

def sample(world,memory):
    out=dict(time_s=float(world.data.time),doors={})
    for oid,description in world.truth.items():
        if description['kind']!='door':continue
        joint=world.model.joint('hinge_'+oid).id;q=int(world.model.jnt_qposadr[joint]);v=int(world.model.jnt_dofadr[joint])
        rec=memory.records[oid]
        out['doors'][oid]=dict(truth_angle_rad=float(world.data.qpos[q]),truth_speed_rad_s=float(world.data.qvel[v]),
                              estimated_angle_rad=rec.angle,estimated_speed_rad_s=rec.angular_velocity,
                              estimate_age_s=float(world.data.time-rec.angle_time) if rec.angle_time>=0 else None)
    return out
