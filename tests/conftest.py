import os
from pathlib import Path
import pytest
import yaml

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute root directory of the implementation project."""
    return Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def skills_config(project_root: Path) -> dict:
    """Loads and returns the parsed .agents/skills.config.yaml dictionary."""
    config_path = project_root / ".agents" / "skills.config.yaml"
    assert config_path.exists(), f"Configuration file not found at {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@pytest.fixture
def expected_benchmark_headers() -> list:
    """Returns the mandated 24-column header list for logs/benchmark_results.csv."""
    return [
        "run_id", "device", "device_mode", "model_variant", "backend",
        "precision", "input_size", "batch", "threads", "map5095",
        "ap50", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "latency_p99_ms", "fps", "power_idle_w", "power_load_w",
        "power_dynamic_w", "fps_per_watt", "joule_per_frame",
        "peak_ram_mb", "temperature_start", "temperature_end"
    ]
