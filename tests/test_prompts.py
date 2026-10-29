"""Prompt coverage, paired task goals, and authored dependency regression checks."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("build_prompts", ROOT / "scripts/build_prompts.py")
BUILDER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(BUILDER)


class PromptConditionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = BUILDER.build(ROOT)
        cls.prompts = {condition: cls.outputs[f"prompts/{condition}.json"] for condition in ("default", "generic", "specific")}
        cls.metadata = cls.outputs["catalog/prompt_conditions.json"]["scenes"]

    def test_all_scene_conditions_cover_catalog_without_duplicates(self):
        catalog = json.loads((ROOT / "catalog/scenes.json").read_text())
        rows = catalog["scenes"] if isinstance(catalog, dict) else catalog
        expected = {row["scene_id"] for row in rows}
        self.assertEqual(92, len(expected))
        self.assertEqual(276, sum(len(prompts) for prompts in self.prompts.values()))
        for condition, prompts in self.prompts.items():
            with self.subTest(condition=condition):
                self.assertEqual(expected, set(prompts))

    def test_shared_task_goal_and_uniform_generic_suffix(self):
        for scene_id, goal in self.prompts["default"].items():
            with self.subTest(scene_id=scene_id):
                self.assertEqual(goal + " " + BUILDER.GENERIC_SUFFIX, self.prompts["generic"][scene_id])
                self.assertEqual(goal + " " + self.metadata[scene_id]["specific_suffix"], self.prompts["specific"][scene_id])
                self.assertNotIn("without disturbing", goal)

    def test_authored_goals_match_current_specifications(self):
        specs = {row['scene_id']: row for family in BUILDER.FAMILIES
                 for row in json.loads((ROOT / f'data/specs/{family}.json').read_text())}
        for scene_id, row in self.metadata.items():
            self.assertEqual(specs[scene_id]['instruction'], row['goal'])
        for scene_id in (f'P{i:02d}' for i in range(21, 28)):
            self.assertIn('flat', self.prompts['default'][scene_id])
        for scene_id in ("O21", "O31", "O32"):
            self.assertIn("passage", self.prompts["default"][scene_id])
        for scene_id in ("P28", "P29", "P30"):
            self.assertIn("each end supported", self.prompts["default"][scene_id])

    def test_ish16_actual_stack_top_down_and_container(self):
        row = self.metadata["ISH-N16"]
        facts = row["risk_facts"]
        self.assertIn({"kind": "supported_by", "object": "chocolate_pudding", "support": "coffee_can"}, facts)
        self.assertIn({"kind": "supported_by", "object": "mug", "support": "chocolate_pudding"}, facts)
        self.assertIn({"kind": "top_down_relocation_order", "objects": ["mug", "chocolate_pudding"]}, facts)
        self.assertIn({"kind": "temporary_container", "object": "bin_a04"}, facts)
        text = row["specific_suffix"]
        self.assertIn("the mug and then the chocolate-pudding box", text)
        self.assertNotIn("white bottle is supported", text)

    def test_c05_leaning_fact_and_surrounding_objects_are_distinct(self):
        facts = self.metadata["C05"]["risk_facts"]
        self.assertIn({"kind": "leaning_against", "object": "smartphone", "support": "coffee_can"}, facts)
        self.assertIn({"kind": "surrounding_objects", "target": "coffee_can", "objects": ["corn_can", "spam_can"]}, facts)
        self.assertIn({"kind": "temporary_container", "object": "bin_a05"}, facts)
        self.assertIn("smartphone has been safely relocated", self.prompts["specific"]["C05"])
        self.assertNotIn("relocated", self.prompts["specific"]["C01"])

    def test_p25_actual_corn_to_jello_stack(self):
        facts = self.metadata["P25"]["risk_facts"]
        self.assertIn({"kind": "supported_by", "object": "stack1_2_jello", "support": "stack1_1_corn_can"}, facts)
        text = self.prompts["specific"]["P25"]
        self.assertIn("A jello box rests on a corn can.", text)
        self.assertIn("mug", text)
        self.assertNotIn("mug rests on", text)

    def test_p26_actual_soup_to_tuna_stack_not_ground_tuna(self):
        facts = self.metadata["P26"]["risk_facts"]
        edges = [fact for fact in facts if fact["kind"] == "supported_by"]
        self.assertEqual([{"kind": "supported_by", "object": "stack1_2_tuna_can", "support": "stack1_1_soup_can"}], edges)
        existing = next(fact["objects"] for fact in facts if fact["kind"] == "existing_bin_objects")
        self.assertIn("existing_2_tuna_can", existing)
        self.assertIn("A tuna can rests on a soup can.", self.prompts["specific"]["P26"])

    def test_p27_two_actual_stacks_and_three_layer_chain(self):
        edges = {(fact["object"], fact["support"]) for fact in self.metadata["P27"]["risk_facts"] if fact["kind"] == "supported_by"}
        self.assertEqual({
            ("stack1_2_tuna_can", "stack1_1_corn_can"),
            ("stack1_3_jello", "stack1_2_tuna_can"),
            ("stack2_2_soup_can", "stack2_1_canned_mushrooms"),
        }, edges)
        text = self.prompts["specific"]["P27"]
        self.assertIn("A tuna can rests on a corn can.", text)
        self.assertIn("A jello box rests on a tuna can.", text)
        self.assertIn("A soup can rests on a mushroom can.", text)

    def test_curve_prompts_preserve_actual_shapes(self):
        for scene_id, shape in (("O31", "C"), ("O32", "S")):
            with self.subTest(scene_id=scene_id):
                self.assertIn({"kind": "passage_shape", "shape": shape}, self.metadata[scene_id]["risk_facts"])
                self.assertIn(f"{shape}-shaped", self.prompts["specific"][scene_id])
                self.assertIn("inside corners", self.prompts["specific"][scene_id])

    def test_bridge_uses_both_actual_support_ids(self):
        for scene_id in ("P28", "P29", "P30"):
            self.assertIn({"kind": "required_bridge_supports", "objects": ["support_left", "support_right"], "unsupported_middle": True}, self.metadata[scene_id]["risk_facts"])
            self.assertIn("blue block and the red block", self.prompts["specific"][scene_id])
            self.assertIn("only one end is supported", self.prompts["specific"][scene_id])

    def test_every_referenced_object_exists_and_no_machine_paths_leak(self):
        for family in BUILDER.FAMILIES:
            specs = json.loads((ROOT / "data/specs" / f"{family}.json").read_text())
            for spec in specs:
                known = set(BUILDER.object_map(spec))
                for fact in self.metadata[spec["scene_id"]]["risk_facts"]:
                    for key in ("object", "support", "target", "container"):
                        if key in fact:
                            self.assertIn(fact[key], known)
                    for obj in fact.get("objects", []):
                        self.assertIn(obj, known)
        serialized = json.dumps(self.outputs)
        for private in ("/shared/", "/home/", "user70", "checkpoint", "is_violation"):
            self.assertNotIn(private, serialized)

    def test_committed_outputs_reproduce_exactly(self):
        for relative, payload in self.outputs.items():
            expected = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            self.assertEqual(expected, (ROOT / relative).read_text(), relative)


if __name__ == "__main__":
    unittest.main()
