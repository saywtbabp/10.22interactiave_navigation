"""Single responsive live control room. Physics and exposure use simulation time.

UI redraw/pacing uses wall time and cannot change rotor commands or decisions.
Prediction work runs in processes while this main-thread Tk/OpenGL UI pumps.
"""
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image,ImageTk
from .art import dashboard,ObserverRenderer,save_views,BG,TEAL,MUTED,WHITE
from preview import overview_camera

class UserClosed(RuntimeError):pass

class LiveUI:
    def __init__(self,speed=1.):
        self.root=tk.Tk();self.root.title('INTERACT  ·  无人机主动交互导航  |  MuJoCo')
        self.root.configure(bg=BG);sw,sh=self.root.winfo_screenwidth(),self.root.winfo_screenheight()
        self.root.geometry('{}x{}+20+15'.format(min(1560,sw-50),min(1080,sh-70)))
        self.root.minsize(1000,760);self.closed=False;self.paused=False;self.done=False;self.mission=None;self.renderer=None
        self.speed=tk.DoubleVar(value=speed);self.last_draw=0.;self.last_sim=0.;self.last_wall=time.perf_counter()
        self.last_dashboard=None;self.photo=None;self.drag=None;self.scale=1.;self.offset=(0,0)
        bar=tk.Frame(self.root,bg='#101f30',height=42);bar.pack(fill='x')
        def button(label,command):
            b=tk.Button(bar,text=label,command=command,bg='#1b3045',fg=WHITE,activebackground='#25566a',
                        activeforeground=WHITE,relief='flat',font=('Microsoft YaHei UI',10),padx=14,pady=5,cursor='hand2')
            b.pack(side='left',padx=(8,0),pady=5);return b
        self.pause_button=button('Ⅱ 暂停 / 空格',self.toggle_pause)
        button('↖ 恢复总览 R',self.reset_camera);button('◎ 跟随无人机 F',self.toggle_follow)
        button('▣ 保存三视图 S',self.save_current)
        tk.Label(bar,text='播放速度',bg='#101f30',fg=MUTED,font=('Microsoft YaHei UI',10)).pack(side='left',padx=(25,8))
        combo=ttk.Combobox(bar,textvariable=self.speed,values=[.25,.5,1.,2.,4.,8.],width=5,state='readonly');combo.pack(side='left')
        self.notice=tk.Label(bar,text='真实物理 · 同步观测',bg='#101f30',fg=TEAL,font=('Microsoft YaHei UI',10));self.notice.pack(side='right',padx=16)
        self.canvas=tk.Canvas(self.root,bg=BG,highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<ButtonPress-1>',self.mouse_down);self.canvas.bind('<B1-Motion>',self.mouse_move);self.canvas.bind('<MouseWheel>',self.zoom)
        self.root.bind('<space>',lambda e:self.toggle_pause());self.root.bind('r',lambda e:self.reset_camera())
        self.root.bind('f',lambda e:self.toggle_follow());self.root.bind('s',lambda e:self.save_current())
        self.root.protocol('WM_DELETE_WINDOW',self.request_close);self.root.update()

    def attach(self,mission):
        self.mission=mission;self.renderer=ObserverRenderer(mission);self.last_sim=mission.world.data.time
        self.last_wall=time.perf_counter();self.draw()

    def toggle_pause(self):
        self.paused=not self.paused;self.pause_button.configure(text='▶ 继续 / 空格' if self.paused else 'Ⅱ 暂停 / 空格')
        self.last_wall=time.perf_counter()

    def reset_camera(self):
        if self.renderer:self.renderer.camera=overview_camera();self.renderer.follow=False;self.last_draw=0

    def toggle_follow(self):
        if self.renderer:
            self.renderer.follow=not self.renderer.follow
            if not self.renderer.follow:self.reset_camera()
            self.last_draw=0

    def local(self,event):return ((event.x-self.offset[0])/self.scale,(event.y-self.offset[1])/self.scale)
    def mouse_down(self,event):
        x,y=self.local(event)
        if 24<x<1032 and 160<y<700:self.drag=(event.x,event.y)

    def mouse_move(self,event):
        if self.drag is None or self.renderer is None:return
        self.renderer.follow=False;dx,dy=event.x-self.drag[0],event.y-self.drag[1];self.drag=(event.x,event.y)
        self.renderer.camera.azimuth-=dx*.3;self.renderer.camera.elevation=max(-89,min(-8,self.renderer.camera.elevation+dy*.2));self.last_draw=0

    def zoom(self,event):
        x,y=self.local(event)
        if self.renderer and 24<x<1032 and 160<y<700:
            self.renderer.camera.distance=max(1.5,min(35,self.renderer.camera.distance*(.90 if event.delta>0 else 1.1)));self.last_draw=0

    def request_close(self):self.closed=True

    def draw(self):
        if self.closed:return
        self.last_dashboard=dashboard(self.mission,self.renderer)
        ww,hh=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height());self.scale=min(ww/1600,hh/1040)
        size=(max(1,int(1600*self.scale)),max(1,int(1040*self.scale)))
        display=self.last_dashboard.resize(size,Image.Resampling.LANCZOS);self.offset=((ww-size[0])//2,(hh-size[1])//2)
        self.photo=ImageTk.PhotoImage(display);self.canvas.delete('all');self.canvas.create_image(*self.offset,anchor='nw',image=self.photo)
        self.notice.configure(text=('物理暂停 · 预测计算 {}/{}'.format(*self.mission.progress) if self.mission.prediction else
                                    '已暂停' if self.paused else '任务完成 · 可旋转检查 / 保存' if self.mission.success else '真实物理 · 同步观测'))
        self.last_draw=time.perf_counter()

    def tick(self,predicting=False):
        if self.closed:raise UserClosed('user_closed_window')
        now=time.perf_counter()
        if self.mission and now-self.last_draw>.12:self.draw()
        self.root.update_idletasks();self.root.update()
        while self.paused and not self.closed:
            if time.perf_counter()-self.last_draw>.15:self.draw()
            self.root.update();time.sleep(.01)
        if self.closed:raise UserClosed('user_closed_window')
        sim=self.mission.world.data.time if self.mission else 0.
        if predicting:
            self.last_sim=sim;self.last_wall=time.perf_counter();return
        duration=(sim-self.last_sim)/max(.1,float(self.speed.get()))
        remaining=duration-(time.perf_counter()-self.last_wall)
        while remaining>.001:
            self.root.update();time.sleep(min(.01,remaining));remaining=duration-(time.perf_counter()-self.last_wall)
        self.last_sim=sim;self.last_wall=time.perf_counter()

    def save(self,prefix):
        save_views(self.mission,prefix,renderer=self.renderer)

    def save_current(self):
        if self.mission:
            self.save(self.mission.folder/('manual_{:08.2f}'.format(self.mission.world.data.time)))

    def keep_open(self):
        self.done=True;self.paused=False
        while not self.closed:
            self.draw();self.root.update();time.sleep(.03)

    def close(self):
        if self.renderer:self.renderer.close();self.renderer=None
        try:self.root.destroy()
        except tk.TclError:pass
        self.closed=True
