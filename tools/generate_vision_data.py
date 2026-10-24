"""Offline supervised data generation. Simulator IDs ONLY label training images.

Runtime perception never imports this script. Splits use different randomized
scene instances, materials, light and poses, not neighboring video frames.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import argparse
import json
import math
import hashlib
import numpy as np
import mujoco
from PIL import Image
from configuration import ROOT, Config, architecture
from navigation.physics import World
from navigation.sensors import Sensors


def pose(world, base, position, yaw, pitch, roll):
    m, d = world.model, world.data
    d.qpos[:] = base
    mujoco.mj_forward(m, d)
    origin = d.sensor('pos').data.copy()
    # Compose world yaw, body pitch and roll; render fixture only.
    q = np.array([1., 0, 0, 0])
    for axis, a in [([0., 0, 1], yaw), ([1., 0, 0], pitch), ([0., 1, 0], roll)]:
        dq = np.r_[math.cos(a/2), np.array(axis)*math.sin(a/2)]
        out = np.empty(4); mujoco.mju_mulQuat(out, q, dq); q = out
    matrix = np.empty(9); mujoco.mju_quat2Mat(matrix, q); R = matrix.reshape(3, 3)
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) in world.robot_ids:
            a = int(m.jnt_qposadr[j]); old = base[a+3:a+7].copy()
            d.qpos[a:a+3] = position + R @ (base[a:a+3]-origin)
            mujoco.mju_mulQuat(d.qpos[a+3:a+7], q, old)
    mujoco.mj_forward(m, d)


def generate(root, scenes=18, views=48):
    arch, cfg = architecture(), Config()
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scene in range(scenes):
        split = 'train' if scene < int(scenes*.70) else 'val' if scene < int(scenes*.85) else 'test'
        rng = np.random.default_rng(9127+scene*71)
        contents = {o['id']: str(rng.choice(['door', 'glass', 'empty'], p=[.42, .38, .20])) for o in arch['openings']}
        world = World(arch, cfg, observed_contents=contents)
        m, d = world.model, world.data
        # Class colors deliberately overlap; color alone cannot solve this data.
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if name.startswith('surface_'):
                oid = name[len('surface_'):]
                if contents.get(oid) == 'door':
                    if scene % 3:
                        m.geom_matid[g] = -1
                        m.geom_rgba[g, :3] = rng.uniform(.12, .82, 3)
                else:
                    mat = int(m.geom_matid[g])
                    m.mat_rgba[mat, :3] = rng.uniform(.25, .85, 3)
                    m.mat_rgba[mat, 3] = rng.uniform(.10, .40)
            elif name.startswith('wall_') and rng.random() < .65:
                m.geom_matid[g] = -1
                m.geom_rgba[g, :3] = rng.uniform(.35, .85, 3)
        m.light_diffuse[:] *= rng.uniform(.65, 1.20)
        base = d.qpos.copy()
        sensor = Sensors(world, cfg)
        renderer = mujoco.Renderer(m, height=cfg.camera_height, width=cfg.camera_width)
        targets = {}
        for oid, kind in contents.items():
            if kind in ('door', 'glass'):
                targets[m.geom('surface_'+oid).id] = 0 if kind == 'door' else 1
        for k in range(views):
            o = arch['openings'][k % len(arch['openings'])]
            x, y = o['center_xy_m']; distance = rng.uniform(.50, 3.15)
            position = np.array([np.clip(x+rng.uniform(-1.10, 1.10), -5.05, 5.05),
                                 max(.45, y-distance), rng.uniform(.95, 1.55)])
            yaw = math.atan2(-(x-position[0]), y-position[1])+rng.uniform(-.25, .25)
            if k % 13 == 0: yaw += rng.choice([-1, 1])*1.15
            pose(world, base, position, yaw, rng.uniform(-.12, .12), rng.uniform(-.10, .10))
            for oid, kind in contents.items():
                if kind == 'door':
                    j = m.joint('hinge_'+oid).id
                    d.qpos[int(m.jnt_qposadr[j])] = 0. if k % 4 else rng.uniform(0, 1.1)
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera='onboard', scene_option=sensor.opt)
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(d, camera='onboard', scene_option=sensor.opt)
            ids = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            labels = []
            for gid, cls in targets.items():
                yy, xx = np.nonzero((ids[:, :, 0] == gid) & (ids[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
                if len(xx) < 120: continue
                x1, x2, y1, y2 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
                if min(x2-x1, y2-y1) < 10: continue
                labels.append([cls, (x1+x2)/(2*cfg.camera_width), (y1+y2)/(2*cfg.camera_height),
                               (x2-x1)/cfg.camera_width, (y2-y1)/cfg.camera_height])
            stem = 'scene_%02d_view_%03d' % (scene, k)
            image_dir, label_dir = root/'images'/split, root/'labels'/split
            image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(image_dir/(stem+'.jpg'), quality=94)
            (label_dir/(stem+'.txt')).write_text('\n'.join('%d %.7f %.7f %.7f %.7f' % tuple(a) for a in labels), encoding='utf-8')
            manifest.append(dict(image='images/%s/%s.jpg' % (split, stem), scene=scene, split=split,
                                 labels=len(labels), seed=9127+scene*71))
        renderer.close(); sensor.close()
        print('rendered scene', scene, split, len(manifest), flush=True)
    (root/'data.yaml').write_text('path: '+root.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: door\n  1: glass\n', encoding='utf-8')
    report = dict(scope='MuJoCo synthetic research domain; no real-world generalization claim',
                  annotation='offline visible simulator segmentation; never supplied to runtime perception',
                  split_unit='randomized scene instance', scenes=scenes, images=len(manifest), entries=manifest)
    (root/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--scenes', type=int, default=18); p.add_argument('--views', type=int, default=48)
    a = p.parse_args(); generate(ROOT/'data'/'vision_synthetic', a.scenes, a.views)
