"""PyCharm: right-click Run, no arguments. All paths are relative to this file."""
import runtime
import argparse
from dataclasses import replace
from multiprocessing import freeze_support
import sys
import mujoco
import numpy as np
import PIL
from configuration import load_config

def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='基于交互的无人机导航 / MuJoCo')
    parser.add_argument('--headless',action='store_true',help='不创建 Tk 窗口；机载 RGB 仍真实渲染')
    parser.add_argument('--variant',choices=['standard','glass_removed','swapped','all_empty'],default='standard')
    parser.add_argument('--baseline',choices=['adaptive','always_avoid','always_push','no_probe'],default='adaptive')
    parser.add_argument('--debug-power',type=int,help='仅开发调试；正式运行默认 10 = 1024')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--close-on-done',action='store_true')
    parser.add_argument('--hard-door',action='store_true')
    args=parser.parse_args();cfg=load_config()
    if args.debug_power is not None:cfg=replace(cfg,sobol_power=args.debug_power)
    print('Interpreter:',sys.executable,flush=True)
    print('Python',sys.version.split()[0],'MuJoCo',mujoco.__version__,'NumPy',np.__version__,'Pillow',PIL.__version__,flush=True)
    print('N =',1<<cfg.sobol_power,'(DEBUG ONLY)' if cfg.sobol_power!=10 else '(formal domain)',flush=True)
    ui=None
    if not args.headless:
        from ui.live import LiveUI
        ui=LiveUI(speed=args.speed)
    from navigation.mission import Mission
    parameters={'O01':(.673,6.317),'O05':(.437,2.473)} if args.hard_door else None
    mission=Mission(cfg,variant=args.variant,baseline=args.baseline,door_parameters=parameters,ui=ui)
    try:
        okay=mission.run()
        if ui and not args.close_on_done:ui.keep_open()
    finally:
        mission.sensors.close()
        if ui:ui.close()
    return 0 if okay else 1

if __name__=='__main__':
    freeze_support();raise SystemExit(main())
