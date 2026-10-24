"""Actual RGB, weak-return glass, missing-vision and physical verification tests.

This is an evaluation fixture: labels and appearance edits stay in this file.
Run with the usual MuJoCo Python; no training environment is required.
"""
import runtime
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import ast
import json
import unittest
import numpy as np
from PIL import Image
from configuration import ROOT, load_config, architecture
from navigation.physics import World
from navigation.sensors import Sensors
from navigation.perception import Memory
from navigation.vision import YoloDetector, VisionFrame
from navigation.force_observer import ForceObserver
from navigation.interactions import approach
from navigation.verification import verify_opening
from experiments.library import save_json

OUTPUT = ROOT/'results'/'sensor_upgrade_validation'


class NoVisualDetector:
    sha = 'explicit-no-visual-test-fixture'
    def infer(self, packet): return VisionFrame(packet, (), 0., self.sha)


class SensorUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.cfg = load_config(); cls.arch = architecture(); cls.detector = YoloDetector(cls.cfg)
        cls.evidence = []

    @classmethod
    def tearDownClass(cls): save_json(OUTPUT/'checks.json', cls.evidence)

    def fixture(self, index=2, variant='standard', visual=True, cfg=None, recolor=None):
        cfg = cfg or self.cfg; arch = deepcopy(self.arch); o = arch['openings'][index]
        arch['start_xyz_m'] = approach(o).tolist(); w = World(arch, cfg, variant=variant)
        if recolor is not None:
            g = w.model.geom('surface_'+o['id']).id
            w.model.geom_matid[g] = -1; w.model.geom_rgba[g] = [*recolor, 1.]
        s = Sensors(w, cfg); memory = Memory(arch, cfg, self.detector if visual else NoVisualDetector())
        state = SimpleNamespace(cfg=cfg, arch=arch, world=w, sensors=s, memory=memory, verifications=[],
                                status='', action='', reason='', metrics={}, trajectory=[], path=np.empty((0, 2)))
        def step(world):
            packet, fresh = s.acquire()
            if fresh: memory.fuse(packet)
            state.trajectory.append([float(world.data.time), *world.observed()[0].tolist()])
        def save(label):
            from ui.art import camera_image, map_image
            camera_image(state).save(OUTPUT/(label+'_camera.png'))
            map_image(state).save(OUTPUT/(label+'_map.png'))
        state.save_key = save; w.callback = step
        w.hold(approach(o), 1.0, stage='INITIAL_SETTLE', yaw=0.)
        return state, memory.records[o['id']]

    def close(self, state): state.memory.close(); state.sensors.close()

    def test_01_real_yolo_default_and_nonbrown_doors(self):
        for label, color in [('walnut', None), ('white', [.78, .78, .78]), ('blue', [.18, .30, .72])]:
            state, rec = self.fixture(0, recolor=color)
            try:
                state.save_key('door_'+label)
                self.assertEqual(rec.semantic, 'door', rec.visual_evidence)
                self.assertIn('YOLO', rec.visual_evidence['source'])
                self.assertEqual(rec.traversability, 'unverified')
                self.evidence.append(dict(case='door_'+label, visual=rec.visual_evidence, passed=True))
            finally: self.close(state)

    def test_02_glass_gap_without_visual_detector(self):
        state, rec = self.fixture(visual=False)
        try:
            self.assertEqual(rec.semantic, 'unknown')
            self.assertEqual(rec.verification_state, 'pending', rec.range_evidence)
            self.assertEqual(rec.traversability, 'unverified')
            self.assertTrue(rec.ambiguity_evidence['ambiguous_gap'])
            self.evidence.append(dict(case='glass_visual_missed', range=rec.ambiguity_evidence, passed=True))
        finally: self.close(state)

    def test_03_real_glass_imu_verification_and_recovery(self):
        state, rec = self.fixture(visual=False)
        try:
            outcome = verify_opening(state, rec)
            save_json(OUTPUT/'glass_verification.json', state.verifications)
            self.assertEqual(outcome, 'blocked', state.verifications[-1])
            self.assertEqual(rec.semantic, 'unknown')
            self.assertGreater(state.world.contact_peak, 0.)  # Independent evaluator only.
            self.assertLess(state.world.wall_peak, .1)
            self.assertIn('IMU', rec.contact_history[-1]['force_source'])
            self.evidence.append(dict(case='glass_imu_contact', outcome=outcome,
                                      estimated_peak=state.verifications[-1]['force_peak_N'], passed=True))
        finally: self.close(state)

    def test_04_empty_aperture_needs_completed_validation(self):
        state, rec = self.fixture(variant='glass_removed', visual=False)
        try:
            self.assertEqual(rec.traversability, 'unverified')
            outcome = verify_opening(state, rec)
            save_json(OUTPUT/'free_verification.json', state.verifications)
            self.assertEqual(outcome, 'free', state.verifications[-1])
            self.assertTrue(rec.contact_history[-1]['valid_validation_position'])
            self.assertLess(state.world.contact_peak, .1)
            self.evidence.append(dict(case='empty_completed', outcome=outcome, passed=True))
        finally: self.close(state)

    def test_05_timeout_is_inconclusive(self):
        cfg = replace(self.cfg, verify_timeout=.15)
        state, rec = self.fixture(variant='glass_removed', visual=False, cfg=cfg)
        try:
            self.assertEqual(verify_opening(state, rec), 'inconclusive')
            self.assertEqual(rec.traversability, 'unverified')
            self.assertEqual(rec.verification_state, 'inconclusive')
            self.evidence.append(dict(case='timeout_not_free', passed=True))
        finally: self.close(state)

    def test_06_invalid_ranges_and_cached_exposures(self):
        state, rec = self.fixture(visual=False)
        try:
            invalid = deepcopy(state.sensors.packet); invalid.time += .1
            invalid.rgb = None; invalid.valid[:] = False; invalid.distances[:] = np.nan; invalid.points[:] = np.nan
            fresh = Memory(self.arch, self.cfg, NoVisualDetector())
            try:
                self.assertTrue(fresh.fuse(invalid))
                self.assertFalse(fresh.fuse(invalid))
                self.assertTrue(all(r.traversability == 'unverified' for r in fresh.records.values()))
                future = deepcopy(state.sensors.packet); future.time += 5.
                self.assertFalse(fresh.fuse_visual(VisionFrame(future, (), 0., 'test'), invalid.time))
                self.assertEqual(fresh.last_visual_frame, -1.)
            finally: fresh.close()
            self.evidence.append(dict(case='invalid_missing_duplicate_future', passed=True))
        finally: self.close(state)

    def test_07_force_observer_hover_contact_and_stale(self):
        f = ForceObserver(self.cfg)
        for i in range(40): f.update(i*.01, [0., 0., 9.81], np.eye(3), 1., 9.81, calibrate=True)
        self.assertLess(np.linalg.norm(f.force-f.bias), 1e-6)
        for i in range(40, 70): f.update(i*.01, [-1., 0., 9.81], np.eye(3), 1., 9.81)
        self.assertGreater(f.normal_force([1., 0.]), .9)
        f.update(2., [0., 0., 9.81], np.eye(3), 1., 9.81)
        self.assertFalse(f.valid)
        self.evidence.append(dict(case='force_observer', passed=True))

    def test_08_no_model_fallback_and_no_truth_imports(self):
        with self.assertRaises(FileNotFoundError): YoloDetector(replace(self.cfg, yolo_model='models/vision/missing_test_model.onnx'))
        for name in ('vision.py', 'ambiguity.py', 'force_observer.py', 'perception.py'):
            tree = ast.parse((ROOT/'navigation'/name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, {'truth', 'qpos', 'qvel', 'geom_xpos', 'geom_group', 'ncon'}, name)
        self.evidence.append(dict(case='runtime_information_boundary', passed=True))

    def test_09_force_sensor_failure_cannot_validate_free(self):
        state, rec = self.fixture(variant='glass_removed', visual=False)
        original=state.sensors.measure_force
        def invalid_force(calibrate=False):
            value=original(calibrate);value['valid']=False;return value
        state.sensors.measure_force=invalid_force
        try:
            self.assertEqual(verify_opening(state,rec),'inconclusive')
            self.assertEqual(rec.traversability,'unverified')
            self.assertEqual(state.verifications[-1]['valid_force_samples'],0)
            self.evidence.append(dict(case='force_failure_not_free',passed=True))
        finally:self.close(state)

    def test_10_contact_memory_survives_semantic_conflict(self):
        from navigation.vision import Detection
        state,rec=self.fixture(0)
        try:
            state.memory.verified(rec.stable_id,1.,'blocked',1.,force_source='explicit contact-history test fixture')
            original=deepcopy(rec.contact_history)
            for k in range(3):
                p=deepcopy(state.sensors.packet);p.time=1.05+k*.1
                detection=Detection(tuple(rec.visual_evidence['bbox_xyxy']),'suspected_glass',.99)
                state.memory.fuse_visual(VisionFrame(p,(detection,),0.,'test'),p.time)
            self.assertEqual(rec.semantic,'door')
            self.assertEqual(rec.traversability,'blocked')
            self.assertEqual(rec.contact_history,original)
            self.assertEqual(rec.confirmation,'pending_verification')
            self.evidence.append(dict(case='contact_memory_conflict',passed=True))
        finally:self.close(state)


if __name__ == '__main__': unittest.main(verbosity=2)
