"""Real Tk/OpenGL run, automated pause/resume, camera controls and replay equality."""
import runtime
from multiprocessing import freeze_support
import sys
import numpy as np
from configuration import ROOT,Config
from navigation.mission import Mission
from experiments.library import save_json

def main():
    from ui.live import LiveUI
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--standard',action='store_true');args=p.parse_args()
    variant='standard' if args.standard else 'all_empty'
    folder=ROOT/'results'/('gui_yolo_consistency_'+variant);folder.mkdir(exist_ok=True)
    headless=Mission(Config(),variant=variant,output=folder/'headless')
    headless.run();headless.memory.close();headless.sensors.close()
    ui=LiveUI(speed=8.);live=Mission(Config(),variant=variant,output=folder/'live',ui=ui)
    checks={}
    def pause():ui.toggle_pause();checks['paused']=ui.paused
    def resume():ui.toggle_pause();checks['resumed']=not ui.paused
    def follow():ui.toggle_follow();checks['follow']=ui.renderer.follow
    def overview():ui.reset_camera();checks['overview']=not ui.renderer.follow
    ui.root.after(350,pause);ui.root.after(850,resume);ui.root.after(1400,follow);ui.root.after(2200,overview)
    if args.standard:
        original_callback=live.world.callback;captured=[False]
        def capture_interaction(world):
            original_callback(world)
            if not captured[0] and world.stage in ('PUSH','PUSH_BRAKE') and live.current:
                rec=live.memory.records[live.current]
                if rec.angle>.52:
                    captured[0]=True;ui.renderer.follow=True;ui.renderer.camera.azimuth=235
                    ui.save(folder/'live_interaction_close');ui.reset_camera()
        live.world.callback=capture_interaction
    try:
        live.run();ui.draw();ui.last_dashboard.save(folder/'live_window_content.png');ui.save(folder/'saved_from_live')
        checks['both_tasks_succeeded']=headless.success and live.success
        checks['telemetry_exactly_equal']=headless.telemetry==live.telemetry
        checks['decisions_equal']=len(headless.decisions)==len(live.decisions) and all(
            (a['opening'],a['action'],a['G'],a['p_D'],a['D'])==(b['opening'],b['action'],b['G'],b['p_D'],b['D'])
            for a,b in zip(headless.decisions,live.decisions))
        checks['simulated_times_equal']=headless.world.data.time==live.world.data.time
        checks['saved_actual_three_views']=(folder/'saved_from_live_3d.png').exists() and (folder/'saved_from_live_map.png').exists() and (folder/'saved_from_live_camera.png').exists()
        if args.standard:checks['saved_interaction_close']=(folder/'live_interaction_close_dashboard.png').exists()
        checks['completion_window_retained']=bool(ui.root.winfo_exists()) and live.success
        ui.root.after(350,ui.request_close);ui.keep_open()
        save_json(folder/'checks.json',checks);print(checks,flush=True)
        assert all(checks.values()),checks
    finally:live.memory.close();live.sensors.close();ui.close()

if __name__=='__main__':freeze_support();main()
