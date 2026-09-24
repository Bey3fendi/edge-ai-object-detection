import unittest

class TestBenchmarkLoggingSchema(unittest.TestCase):
    def setUp(self):
        self.expected_headers = [
            "run_id", "device", "device_mode", "model_variant", "backend",
            "precision", "input_size", "batch", "threads", "map5095",
            "ap50", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
            "latency_p99_ms", "fps", "power_idle_w", "power_load_w",
            "power_dynamic_w", "fps_per_watt", "joule_per_frame",
            "peak_ram_mb", "temperature_start", "temperature_end"
        ]

    def test_benchmark_schema_column_count(self):
        self.assertEqual(len(self.expected_headers), 24)
        self.assertEqual(self.expected_headers[0], "run_id")
        self.assertIn("map5095", self.expected_headers)
        self.assertIn("fps", self.expected_headers)
        self.assertIn("power_dynamic_w", self.expected_headers)
        self.assertIn("fps_per_watt", self.expected_headers)
        self.assertIn("joule_per_frame", self.expected_headers)

    def test_telemetry_metric_calculations(self):
        fps = 30.0
        p_load = 5.0
        p_idle = 2.0

        p_dynamic = p_load - p_idle
        self.assertAlmostEqual(p_dynamic, 3.0, places=3)

        fps_per_watt = fps / p_load
        self.assertAlmostEqual(fps_per_watt, 6.0, places=3)

        joule_per_frame = p_load / fps
        self.assertAlmostEqual(joule_per_frame, 0.166667, places=4)

if __name__ == "__main__":
    unittest.main()
