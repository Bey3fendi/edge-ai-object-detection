import unittest
from pathlib import Path
import yaml

class TestSkillsConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.config_path = cls.project_root / ".agents" / "skills.config.yaml"
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.skills_config = yaml.safe_load(f)

    def test_project_configuration_basics(self):
        project = self.skills_config.get("project", {})
        self.assertEqual(project.get("name"), "edge-yolox-optimization")
        self.assertEqual(project.get("canonical_model"), "yolox-nano")
        self.assertEqual(project.get("input_resolution"), [1, 3, 416, 416])
        self.assertEqual(project.get("coco_classes"), 80)
        self.assertEqual(project.get("output_channels"), 85)

    def test_dataset_paths_defined(self):
        dataset = self.skills_config.get("dataset", {})
        self.assertIn("train_full", dataset)
        self.assertIn("val_full", dataset)
        self.assertGreater(dataset.get("calibration_size", 0), 0)

    def test_nas_and_pruning_constraints(self):
        nas = self.skills_config.get("nas", {})
        self.assertGreater(nas.get("candidates_count", 0), 0)
        self.assertIn(416, nas.get("resolutions", []))

        pruning = self.skills_config.get("pruning", {})
        self.assertEqual(pruning.get("method"), "structured_channel")
        self.assertTrue(0.0 < pruning.get("default_sparsity", 0.0) < 1.0)

    def test_quantization_specifications(self):
        quant = self.skills_config.get("quantization", {})
        self.assertEqual(quant.get("format"), "QDQ")
        self.assertEqual(quant.get("activation_type"), "QUInt8")
        self.assertEqual(quant.get("weight_type"), "QInt8")
        self.assertEqual(quant.get("accuracy_drop_threshold_map"), 1.0)

if __name__ == "__main__":
    unittest.main()
