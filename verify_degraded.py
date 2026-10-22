"""End-to-end navigation with actual missing RGB frames and extra invalid ranges.

Uses the explicit Always-Avoid baseline to isolate perception/navigation from
new state-specific candidate construction; not an adaptive-policy benchmark.
"""
import runtime
from configuration import ROOT,load_config
from navigation.mission import Mission
from experiments.library import save_json

def main():
    folder=ROOT/'results'/'degraded_sensing_c025_r025'
    mission=Mission(load_config(),baseline='always_avoid',output=folder)
    mission.sensors.camera_dropout=.25;mission.sensors.range_dropout=.25
    try:
        success=mission.run()
        result=dict(success=success,camera_dropout=.25,extra_range_dropout=.25,
                    dropped_rgb_frames=mission.sensors.dropout_frames,total_sensor_frames=mission.sensors.frames,
                    simulated_time_s=mission.world.data.time,scope='explicit Always-Avoid baseline; real sensor measurement outages')
        save_json(folder/'degraded_checks.json',result);print(result,flush=True)
        assert success and mission.sensors.dropout_frames>0,result
    finally:mission.sensors.close()

if __name__=='__main__':main()
