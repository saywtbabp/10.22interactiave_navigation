"""Render actual MuJoCo state, never a concept image."""
import runtime
import mujoco
from PIL import Image
from configuration import ROOT,Config,architecture
from navigation.physics import World

def overview_camera():
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[0,6.3,.25];cam.distance=20.8;cam.azimuth=65;cam.elevation=-59
    return cam

def run():
    w=World(architecture(),Config());w.hold(w.target,.8)
    renderer=mujoco.Renderer(w.model,height=1080,width=1440)
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.geomgroup[:]=1
    renderer.update_scene(w.data,camera=overview_camera(),scene_option=opt)
    Image.fromarray(renderer.render()).save(ROOT/'results'/'scene_preview.png')
    opt.geomgroup[2]=0;opt.geomgroup[5]=0;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=0
    renderer.update_scene(w.data,camera='onboard',scene_option=opt)
    Image.fromarray(renderer.render()).save(ROOT/'results'/'onboard_preview.png')
    renderer.close();print(ROOT/'results'/'scene_preview.png')

if __name__=='__main__':run()
