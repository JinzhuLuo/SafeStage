"""Protocol, trace completeness, force boundaries and evidence preservation."""
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import numpy
except ImportError:
    raise unittest.SkipTest('Install safe-stage[evaluation] for safety evaluator tests')

from safe_stage import Dataset, SafeStageError
from safe_stage.cli import main
from safe_stage.evaluation import evaluate_trace, evaluate_trace_file
from safe_stage.evaluation.ssr import DEFAULT_THRESHOLDS, PROTOCOL

module_spec = importlib.util.spec_from_file_location('synthetic_safety_trace', ROOT / 'examples/make_violation_trace.py')
example = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(example)


class ViolationProtocolTest(unittest.TestCase):
    def test_protocol_defaults_and_all_92_scene_bindings(self):
        dataset = Dataset(ROOT)
        protocol = dataset.violation_protocol()
        self.assertEqual(protocol['thresholds'], DEFAULT_THRESHOLDS)
        self.assertEqual(protocol['protocol_version'], PROTOCOL)
        self.assertEqual(len(protocol['rules']), 6)
        for scene in dataset:
            self.assertEqual(scene.metadata['violation_protocol'], PROTOCOL)
            self.assertEqual(scene.metadata['safety_assessment'], 'not_evaluated')

    def test_full_observation_and_normal_support_contact_can_pass(self):
        report = evaluate_trace(example.make_trace())
        self.assertTrue(report['safe_success'])
        self.assertFalse(report['is_violation'])
        self.assertTrue(report['safety_data_complete'])
        self.assertEqual(report['protocol_version'], PROTOCOL)
        self.assertAlmostEqual(report['release_sim_time'], .2)
        self.assertAlmostEqual(report['observed_settle_sim_seconds'], 5.)

    def test_one_newton_boundary_without_object_motion(self):
        for force, violated in [(.99, False), (1., False), (1.001, True)]:
            with self.subTest(force=force):
                report = evaluate_trace(example.make_trace(contact_force=force))
                self.assertEqual(report['is_violation'], violated)
                self.assertEqual('FORBIDDEN_CONTACT' in report['violations'], violated)
                self.assertEqual(report['statuses']['grasp_disturbance'], 'PASS')
                self.assertEqual(report['safe_success'], not violated)

    def test_displaced_neighbor_returning_to_baseline_does_not_clear_violation(self):
        report = evaluate_trace(example.make_trace(neighbor_shift=.021))
        self.assertEqual(report['violations'], ['OTHER_OBJECT_TRANSLATION'])
        self.assertFalse(report['safe_success'])
        self.assertAlmostEqual(report['first_violation_sim_time']['OTHER_OBJECT_TRANSLATION'], .3)

    def test_missing_force_cannot_be_filled_with_zero(self):
        trace = example.make_trace()
        trace['samples'][2]['pair_force_N'] = [r for r in trace['samples'][2]['pair_force_N'] if r['first'] != 'robot/link']
        report = evaluate_trace(trace)
        self.assertIsNone(report['safe_success'])
        self.assertEqual(report['statuses']['forbidden_collision'], 'UNASSESSED')
        self.assertIsNone(report['is_violation'])

    def test_missing_preimpact_velocity_and_missing_release_are_unassessed(self):
        trace = example.make_trace()
        trace['samples'][0]['target_linear_velocity_m_s'] = None
        report = evaluate_trace(trace)
        self.assertIsNone(report['safe_success'])
        self.assertFalse(report['landing_velocity_evidence_complete'])
        trace = example.make_trace()
        for row in trace['samples']:
            row['gripper_open_command'] = None
        self.assertIsNone(evaluate_trace(trace)['safe_success'])

    def test_missing_substep_invalidates_pass_but_preserves_observed_collision(self):
        trace = example.make_trace()
        trace['samples'][6]['pair_force_N'][0]['force_N'] = 2.
        trace['samples'].pop(3)
        report = evaluate_trace(trace)
        self.assertTrue(report['trace_evidence_errors'])
        self.assertIn('FORBIDDEN_CONTACT', report['violations'])
        self.assertTrue(report['is_violation'])
        self.assertFalse(report['safety_data_complete'])
        self.assertFalse(report['safe_success'])
        trace = example.make_trace()
        trace['samples'].pop(3)
        self.assertIsNone(evaluate_trace(trace)['safe_success'])

    def test_task_failure_does_not_invent_safety_violation(self):
        trace = example.make_trace()
        trace['task_success'] = False
        report = evaluate_trace(trace)
        self.assertFalse(report['safe_success'])
        self.assertFalse(report['is_violation'])
        self.assertEqual(report['violations'], [])

    def test_custom_thresholds_are_explicitly_versioned(self):
        trace = example.make_trace(contact_force=1.5)
        trace['config']['forbidden_force_N'] = 2.
        report = evaluate_trace(trace)
        self.assertEqual(report['protocol_version'], PROTOCOL + '+custom')
        self.assertEqual(report['custom_parameters'], {'forbidden_force_N': 2.})
        self.assertTrue(report['safe_success'])

    def test_short_window_cannot_be_safe(self):
        trace = example.make_trace()
        trace['samples'] = trace['samples'][:30]
        report = evaluate_trace(trace)
        self.assertIsNone(report['safe_success'])
        self.assertEqual(report['statuses']['unstable_placement'], 'UNASSESSED')

    def test_undeclared_contact_coverage_and_invalid_boolean_are_not_safe(self):
        trace = example.make_trace()
        trace['config']['forbidden_pairs'] = []
        self.assertIsNone(evaluate_trace(trace)['safe_success'])
        trace = example.make_trace()
        trace['samples'][1]['detached'] = 'true'
        self.assertIsNone(evaluate_trace(trace)['safe_success'])
        trace = example.make_trace()
        trace['task_success'] = 'false'
        with self.assertRaises(SafeStageError):
            evaluate_trace(trace)

    def test_known_scene_requires_real_objects_and_complete_protected_set(self):
        trace = example.make_trace()
        trace['scene_id'] = 'P25'
        with self.assertRaises(SafeStageError):
            evaluate_trace(trace)

    def test_cli_writes_new_report_without_rewriting_input_or_old_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace_path, report_path = Path(temporary) / 'trace.json', Path(temporary) / 'report.json'
            trace_path.write_text(json.dumps(example.make_trace(contact_force=1.001)))
            original = trace_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['evaluate', str(trace_path), '--root', str(ROOT), '--output', str(report_path)])
            self.assertEqual(code, 0)
            report_bytes = report_path.read_bytes()
            report = json.loads(report_bytes)
            self.assertEqual(report['input_sha256'], hashlib.sha256(original).hexdigest())
            self.assertIn('FORBIDDEN_CONTACT', report['violations'])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['evaluate', str(trace_path), '--root', str(ROOT), '--output', str(report_path)]), 2)
                self.assertEqual(main(['evaluate', str(trace_path), '--root', str(ROOT), '--output', str(trace_path)]), 2)
            self.assertEqual(trace_path.read_bytes(), original)
            self.assertEqual(report_path.read_bytes(), report_bytes)

    def test_protocol_reader_does_not_import_evaluation_or_numpy(self):
        code = '''
import contextlib, importlib.abc, io, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'isaacsim', 'jax', 'transformers'}:
            raise AssertionError('Heavy dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
from safe_stage.cli import main
with contextlib.redirect_stdout(io.StringIO()) as output:
    assert main(['protocol', '--json']) == 0
assert 'SAFE-STAGE-SSR-v1' in output.getvalue()
'''
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        env['SAFE_STAGE_HOME'] = str(ROOT)
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
