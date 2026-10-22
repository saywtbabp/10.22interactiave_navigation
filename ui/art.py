"""Observer-only compositing. Camera pixels are never synthesized from this UI."""
from pathlib import Path
import math
import runtime
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from navigation.geometry import wall_boxes
from preview import overview_camera

BG='#09121e';CARD='#111f30';BORDER='#26364a';WHITE='#e8f1f7';MUTED='#8b9eb4';TEAL='#58e3d1';AMBER='#ffbd78'
COLORS={'unknown':'#7f91a9','door':'#f7b66d','suspected_glass':'#6ebeff','free':'#5ee0b9','blocked':'#ff7f8b'}
FONT=Path('C:/Windows/Fonts/msyh.ttc');LATIN=Path('C:/Windows/Fonts/segoeui.ttf')
def font(size,bold=False):
    return ImageFont.truetype(str(FONT if FONT.exists() else LATIN),size)

def text(draw,xy,value,size=14,fill=WHITE):draw.text(xy,str(value),font=font(size),fill=fill)

def map_image(mission,width=520,height=544):
    im=Image.new('RGB',(width,height),CARD);d=ImageDraw.Draw(im)
    arch=mission.arch;padx,pady=38,28;scale=min((width-2*padx)/11,(height-2*pady)/13)
    ox=(width-11*scale)/2;oy=(height-13*scale)/2
    def xy(p):return (ox+(p[0]+5.5)*scale,height-oy-p[1]*scale)
    for x in np.arange(-5.5,5.51,.5):d.line([xy([x,0]),xy([x,13])],fill='#192b3e',width=1)
    for y in np.arange(0,13.01,.5):d.line([xy([-5.5,y]),xy([5.5,y])],fill='#192b3e',width=1)
    for name,c,h in wall_boxes(arch):
        a,b=xy(c-h),xy(c+h);d.rectangle((a[0],b[1],b[0],a[1]),fill='#91a0ad')
    for oid,rec in mission.memory.records.items():
        x,y=rec.geometry['center_xy_m'];w=rec.geometry['width_m'];color=COLORS.get(rec.semantic,COLORS['unknown'])
        if rec.traversability in ('free','blocked'):color=COLORS[rec.traversability]
        a,b=xy([x-w/2,y]),xy([x+w/2,y])
        if rec.confirmation=='candidate':
            for start in range(int(a[0]),int(b[0]),9):d.line([(start,a[1]),(min(start+5,b[0]),a[1])],fill=color,width=2)
        else:d.line([a,b],fill=color,width=4)
        text(d,(a[0]+1,a[1]-29),oid,10,color)
        semantic={'unknown':'待识别','door':'门','suspected_glass':'疑似玻璃'}[rec.semantic]
        passage={'unverified':'待验证','free':'可通行','blocked':'受阻'}[rec.traversability]
        label=semantic+' / '+passage+(' · 绕行' if rec.bypass_selected else '')
        text(d,(a[0]+1,a[1]-16),label,8,color)
        if rec.semantic=='door' and rec.angle_time>=0:
            from navigation.interactions import panel_geometry
            pivot,width,panel=panel_geometry(rec.geometry,rec.angle)
            end=pivot+width*np.array([math.cos(rec.angle),math.sin(rec.angle)])
            d.line([xy(pivot),xy(end)],fill=color,width=3)
    p=mission.sensors.packet
    if p is not None:
        for pt in p.points[p.valid, :2]:
            if -5.5<=pt[0]<=5.5 and 0<=pt[1]<=13:
                u,v=xy(pt);d.ellipse([u-1,v-1,u+1,v+1],fill='#3f6978')
        loc=p.position;forward=p.rotation[:,1][:2];right=p.rotation[:,0][:2]
        fan=[xy(loc),xy(loc[:2]+(forward-right*.9)*1.6),xy(loc[:2]+(forward+right*.9)*1.6)]
        d.polygon(fan,outline='#2d777a')
    if len(mission.path)>1:d.line([xy(p) for p in mission.path],fill='#dbaa65',width=2)
    if len(mission.trajectory)>1:d.line([xy(t[1:3]) for t in mission.trajectory[::3]],fill=TEAL,width=2)
    for label,point,col in [('S',arch['start_xyz_m'],TEAL),('T',arch['goal_xyz_m'],AMBER)]:
        x,y=xy(point);d.ellipse((x-5,y-5,x+5,y+5),fill=col);text(d,(x+9,y-10),label,13,col)
    pos,vel,R,g=mission.world.observed();c=xy(pos);f=R[:,1][:2];r=R[:,0][:2]
    verts=[xy(pos[:2]+f*.24),xy(pos[:2]-f*.15+r*.15),xy(pos[:2]-f*.08),xy(pos[:2]-f*.15-r*.15)]
    d.polygon(verts,fill=TEAL,outline=WHITE);d.ellipse((c[0]-10,c[1]-10,c[0]+10,c[1]+10),outline=TEAL,width=1)
    return im

def camera_image(mission):
    packet=mission.sensors.packet
    if packet is None or packet.rgb is None:
        im=Image.new('RGB',(480,320),'#142335');d=ImageDraw.Draw(im);text(d,(15,140),'等待有效机载图像',18,MUTED);return im
    im=Image.fromarray(packet.rgb).copy();d=ImageDraw.Draw(im)
    for det in mission.memory.detections:
        col=COLORS.get(det['class_name'],COLORS['unknown']);box=det['bbox_xyxy'];d.rectangle(box,outline=col,width=2)
        label='{} {} {:.2f}'.format(det['stable_id'],det['class_name'],det['score']);boxy=max(0,box[1]-19)
        d.rectangle([box[0],boxy,min(im.width,box[0]+len(label)*7),boxy+18],fill='#101e2d');text(d,(box[0]+3,boxy),label,10,col)
    d.rectangle([0,im.height-22,im.width,im.height],fill='#0a1522');text(d,(8,im.height-20),'BODY RGB  ·  t = {:.2f} s'.format(packet.time),12,TEAL)
    return im

class ObserverRenderer:
    def __init__(self,mission):
        self.mission=mission;self.camera=overview_camera();self.renderer=mujoco.Renderer(mission.world.model,height=540,width=1008)
        self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=1;self.opt.sitegroup[:]=0
        self.follow=False

    def render(self):
        w=self.mission.world
        if self.follow:
            self.camera.lookat[:]=w.observed()[0];self.camera.distance=4.0;self.camera.elevation=-36
        self.renderer.update_scene(w.data,camera=self.camera,scene_option=self.opt)
        # Observer trail: scene additions only. Never model/data or onboard renderer.
        scn=self.renderer.scene;traj=self.mission.trajectory
        for a,b in zip(traj[-700::6],traj[-694::6]):
            if scn.ngeom>=scn.maxgeom:break
            geom=scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.20,.95,.84,.7]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.009,np.array(a[1:4]),np.array(b[1:4]));scn.ngeom+=1
        return Image.fromarray(self.renderer.render().copy())

    def close(self):self.renderer.close()

def dashboard(mission,renderer):
    im=Image.new('RGB',(1600,1040),BG);d=ImageDraw.Draw(im)
    d.rounded_rectangle((24,25,68,69),radius=12,fill=TEAL);text(d,(33,29),'↗',27,BG)
    text(d,(84,22),'INTERACT',30);text(d,(86,64),'自主交互导航  /  MuJoCo 实时实验室',13,MUTED)
    text(d,(630,31),mission.status,23,TEAL if mission.success or mission.prediction else WHITE)
    text(d,(632,67),mission.reason[:62],12,MUTED)
    t=mission.world.data.time;text(d,(1420,26),'{:06.1f} s'.format(t),27);text(d,(1419,65),'SIMULATION CLOCK',10,MUTED)
    for rect in [(24,116,1032,700),(1056,116,1576,700),(24,724,520,1016),(544,724,1040,1016),(1064,724,1576,1016)]:
        d.rounded_rectangle(rect,radius=12,fill=CARD,outline=BORDER,width=1)
    text(d,(42,128),'01   三维物理现场',15);text(d,(770,131),'拖动旋转  ·  滚轮缩放  ·  R 总览',11,MUTED)
    scene=renderer.render();im.paste(scene,(24,160))
    # Redraw lower scene edge so all cards retain rounded visual boundaries.
    text(d,(1074,128),'02   在线认知地图',15);text(d,(1390,131),'v{:03d}'.format(mission.memory.version),12,TEAL)
    im.paste(map_image(mission,500,490),(1066,160))
    for i,(lab,col) in enumerate([('未知',COLORS['unknown']),('门',COLORS['door']),('疑似玻璃',COLORS['suspected_glass']),('净空',COLORS['free']),('受阻',COLORS['blocked'])]):
        xx=1074+i*94;d.ellipse((xx,667,xx+6,673),fill=col);text(d,(xx+10,660),lab,11,MUTED)
    text(d,(42,736),'03   固定机载相机  /  RGB + 检测',14)
    cam=ImageOps.contain(camera_image(mission),(464,235),Image.Resampling.LANCZOS)
    im.paste(cam,(40+(464-cam.width)//2,772))
    text(d,(562,736),'决策与执行',14)
    def metric(x,y,label,val,col=WHITE):text(d,(x,y),label,11,MUTED);text(d,(x,y+22),val,23,col)
    metric(564,779,'几何收益 G', '—' if mission.metrics['G'] is None else '{:.2f} m'.format(mission.metrics['G']),AMBER)
    metric(731,779,'Door 支持 p_D','—' if mission.metrics['p_D'] is None else '{:.3f}'.format(mission.metrics['p_D']))
    metric(892,779,'分歧 D','—' if mission.metrics['D'] is None else '{:.3f}'.format(mission.metrics['D']),TEAL)
    d.line((564,853,1018,853),fill=BORDER)
    text(d,(564,867),'{}  /  {}'.format(mission.current or 'S → T',mission.action),17,TEAL)
    text(d,(564,900),'{}  ·  重规划 {}  ·  Probe {}'.format(mission.world.stage,mission.replans,mission.probes),12,MUTED)
    text(d,(564,929),'外接触 {:.2f} N   |   高度 {:.3f} m'.format(mission.world.force,mission.world.observed()[0][2]),12,MUTED)
    if mission.prediction:
        done,total=mission.progress;d.rounded_rectangle((564,967,1018,975),radius=3,fill=BORDER)
        d.rounded_rectangle((564,967,565+int(453*done/max(1,total)),975),radius=3,fill=TEAL)
        text(d,(564,984),'物理暂停 · 候选计算 {}/{}'.format(done,total),10,TEAL)
    else:text(d,(564,968),'旋翼驱动  ·  真实接触  ·  同步传感器',11,MUTED)
    text(d,(1082,736),'物理候选信念 · '+(mission.belief_owner or '待观测'),14);text(d,(1400,739),'N = {}'.format(len(mission.weights)),12,TEAL)
    theta=mission.theta;weights=mission.weights;maxw=max(float(weights.max()),1e-10)
    x0,y0,ww,hh=1100,786,443,155
    d.rectangle((x0,y0,x0+ww,y0+hh),fill='#0a1624')
    for k,(I,f) in enumerate(theta):
        xx=x0+int((I-mission.cfg.inertia_min)/(mission.cfg.inertia_max-mission.cfg.inertia_min)*(ww-4))
        yy=y0+hh-4-int((f-mission.cfg.friction_min)/(mission.cfg.friction_max-mission.cfg.friction_min)*(hh-4))
        power=math.sqrt(float(weights[k])/maxw);base=np.array([32,56,73]);highlight=np.array([88,227,209])
        color=tuple(np.uint8(base+(highlight-base)*power))
        d.rectangle((xx,yy,xx+3,yy+3),fill=color)
    text(d,(1100,950),'铰链惯量 I  →',10,MUTED);text(d,(1390,950),'τ_f ↑',10,MUTED)
    ess=1/float(np.sum(weights**2));text(d,(1100,976),'有效样本数 {:.1f}   ·   初始等权 / 一次软更新'.format(ess),11,MUTED)
    return im

def save_views(mission,prefix,renderer=None):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    owns_renderer=renderer is None
    if owns_renderer:renderer=ObserverRenderer(mission)
    try:
        renderer.render().save(str(prefix)+'_3d.png')
        map_image(mission,650,760).save(str(prefix)+'_map.png')
        camera_image(mission).save(str(prefix)+'_camera.png')
        dashboard(mission,renderer).save(str(prefix)+'_dashboard.png')
    finally:
        if owns_renderer:renderer.close()
