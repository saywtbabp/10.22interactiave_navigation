"""Physical cruise turns and persistent, exposure-associated visual evidence."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from configuration import ROOT, Config, architecture
from navigation.physics import World
from navigation.sensors import Sensors
from navigation.perception import Memory
from navigation.vision import YoloDetector, VisionFrame
from navigation.interactions import approach
from navigation.mission import Mission
from experiments.library import save_json


class NoVisualDetector:
    sha='heading-test-no-visual'
    def infer(self,packet):return VisionFrame(packet,(),0.,self.sha)


class HeadingTests(unittest.TestCase):
    report={}

    @classmethod
    def tearDownClass(cls):
        folder=ROOT/'results'/'heading_validation';folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/'checks.json',cls.report)

    def test_01_rotor_turns_and_forward_cruise_in_four_directions(self):
        world=World(architecture(),Config(),variant='all_empty',visual=False)
        samples=[];turn_samples=[]
        def sample(w):
            p,v,R,g=w.observed()
            if w.stage=='ALIGN_PATH':turn_samples.append(float(w.data.time))
            # Near a stop the position controller can briefly backtrack to
            # correct overshoot; its gaze deliberately does not flip by 180 deg.
            if w.stage!='FLIGHT' or np.linalg.norm(v[:2])<.15 or np.linalg.norm(w.target[:2]-p[:2])<.15:return
            f=R[:2,1]/np.linalg.norm(R[:2,1]);direction=v[:2]/np.linalg.norm(v[:2])
            samples.append(math.degrees(math.acos(float(np.clip(f@direction,-1,1)))))
        world.callback=sample;world.hold(world.arch['start_xyz_m'],1.2,yaw=0.)
        for point in [(-2.8,1.2),(-2.8,1.7),(-3.8,1.7),(-3.8,1.2)]:
            world.fly([point])
            self.assertLess(np.linalg.norm(world.observed()[0][:2]-point),.065)
        self.assertGreater(len(turn_samples),50)
        self.assertGreater(len(samples),50)
        self.assertLess(max(samples),12.)
        self.assertLess(world.wall_peak,.1)
        self.report['four_direction_flight']=dict(samples=len(samples),max_cruise_error_deg=max(samples),
            wall_peak_N=world.wall_peak,simulated_time_s=float(world.data.time))

    def test_02_partial_evidence_survives_physical_turn_and_no_detections(self):
        cfg=Config();arch=architecture();o=arch['openings'][0]
        arch['start_xyz_m']=approach(o).tolist()
        world=World(arch,cfg);sensors=Sensors(world,cfg)
        memory=Memory(arch,cfg,NoVisualDetector());detector=YoloDetector(cfg)
        rec=memory.records[o['id']]
        def exposure():
            world.hold(approach(o),.1,yaw=0.)
            packet,_=sensors.acquire();memory.ranges.add(packet)
            # Give the real detector this exposure directly. The no-visual
            # worker isolates the deliberate missing-detection interval below.
            frame=detector.infer(packet)
            self.assertTrue(memory.fuse_visual(frame,packet.time))
            return packet,frame
        try:
            world.hold(approach(o),1.,yaw=0.)
            exposure();packet,frame=exposure()
            self.assertEqual(rec.evidence_count,2)
            self.assertEqual(rec.semantic,'unknown')
            def sample(w):
                packet,fresh=sensors.acquire()
                if fresh:memory.fuse(packet)
            world.callback=sample
            world.align_yaw(math.pi/2,stage='ALIGN_OPENING')
            world.align_yaw(0.,stage='ALIGN_OPENING')
            self.assertEqual(rec.evidence_count,2)
            world.callback=None
            packet,frame=exposure()
            self.assertEqual(rec.semantic,'door')
            self.assertEqual(rec.evidence_count,3)
            self.assertFalse(memory.fuse_visual(frame,packet.time))
            self.assertEqual(rec.evidence_count,3)
            # Now use all real detector results and measured scan points.
            memory.close();memory=Memory(arch,cfg,detector)
            world.callback=sample
            world.hold(approach(o),.8,yaw=0.)
            rec=memory.records[o['id']]
            self.assertEqual(rec.semantic,'door')
            self.assertLessEqual(world.data.time-rec.range_evidence['time_s'],.25)
            before_count=rec.evidence_count;before_time=world.data.time
            state=SimpleNamespace(world=world,status='',action='',reason='')
            self.assertTrue(Mission.observe_opening(state,rec))
            self.assertAlmostEqual(world.data.time,before_time)
            self.assertEqual(rec.evidence_count,before_count)
            # Expired current range evidence is refreshed, without resetting
            # the known class or demanding another three visual detections.
            rec.range_evidence['time_s']=world.data.time-1.
            self.assertTrue(Mission.observe_opening(state,rec))
            self.assertLessEqual(world.data.time-rec.range_evidence['time_s'],.25)
            self.assertEqual(rec.semantic,'door')
            self.assertGreaterEqual(rec.evidence_count,before_count)
            # Lack of a current measurement cannot authorize an interaction,
            # but it also cannot erase the static semantic memory.
            world.callback=None;rec.range_evidence['time_s']=world.data.time-1.
            self.assertFalse(Mission.observe_opening(state,rec))
            self.assertEqual(rec.semantic,'door')
            self.assertGreaterEqual(rec.evidence_count,before_count)
            self.report['persistent_evidence']=dict(partial_detections_survived_turn=True,
                third_valid_detection_confirmed=True,duplicate_rejected=True,
                ready_observation_wait_s=0.,stale_range_refreshed=True,missing_current_data_not_authorized=True,
                wall_peak_N=world.wall_peak)
        finally:memory.close();sensors.close()

    def test_03_observation_position_avoids_previously_opened_leaf(self):
        arch=architecture();cfg=Config()
        records={o['id']:SimpleNamespace(geometry=o,semantic='unknown',traversability='unverified',
                 bypass_selected=False,angle=0.) for o in arch['openings']}
        records['O05'].semantic='door';records['O05'].traversability='free';records['O05'].angle=math.radians(88)
        events=[]
        mission=Mission.__new__(Mission);mission.arch=arch;mission.cfg=cfg
        mission.memory=SimpleNamespace(records=records,event=lambda *a,**k:events.append((a,k)))
        mission.world=SimpleNamespace(observed=lambda:(np.array([1.74,9.9,1.25]),None,None,None),
                                      data=SimpleNamespace(time=100.))
        opening=records['O07'].geometry;planner=mission.planner()
        self.assertTrue(planner.collides(approach(opening))[0])
        target=mission.observation_target(opening)
        self.assertFalse(planner.collides(target)[0])
        path,_=planner.plan(mission.world.observed()[0],target)
        self.assertTrue(all(planner.clear(a,b) for a,b in zip(path[:-1],path[1:])))
        self.assertEqual(events[0][0][2],'observation_position_adjusted')
        self.report['occupied_observation_position']=dict(nominal_position_m=approach(opening).tolist(),
                                                        selected_position_m=target.tolist(),path_clear=True)


if __name__=='__main__':unittest.main(verbosity=2)
