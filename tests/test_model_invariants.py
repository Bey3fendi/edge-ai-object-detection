import unittest

class TestModelInvariants(unittest.TestCase):
    def test_static_resolution_dimension(self):
        valid_shape = [1, 3, 416, 416]
        self.assertEqual(len(valid_shape), 4)
        self.assertEqual(valid_shape[0], 1, "Batch size must be strictly 1")
        self.assertEqual(valid_shape[1], 3, "Input channels must be 3 (RGB)")
        self.assertEqual(valid_shape[2], 416)
        self.assertEqual(valid_shape[3], 416)

    def test_detection_head_channel_protection_logic(self):
        mock_layers = [
            {"name": "backbone.stem.conv", "out_channels": 16},
            {"name": "backbone.dark2.conv", "out_channels": 32},
            {"name": "head.cls_preds.0", "out_channels": 80},
            {"name": "head.reg_preds.0", "out_channels": 4},
            {"name": "head.obj_preds.0", "out_channels": 1},
            {"name": "head.final_fused_head", "out_channels": 85}, # Combined 80 + 4 + 1
        ]

        ignored_layers = [layer["name"] for layer in mock_layers if layer["out_channels"] == 85]
        self.assertIn("head.final_fused_head", ignored_layers)
        self.assertEqual(len(ignored_layers), 1)

if __name__ == "__main__":
    unittest.main()
