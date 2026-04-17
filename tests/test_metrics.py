from pathlib import Path
from unittest import TestCase, mock

from benchmark.metrics import MetricsCollector


class MetricsCollectorTests(TestCase):
    def test_hardware_metrics_disabled_by_default(self) -> None:
        collector = MetricsCollector(Path("/tmp/metrics.csv"))

        with mock.patch("benchmark.metrics._sample_powermetrics") as sample_mock:
            row = collector._sample()

        sample_mock.assert_not_called()
        self.assertIsNone(row["gpu_utilization_percent"])
        self.assertIsNone(row["gpu_power_mw"])
        self.assertIsNone(row["die_temp_c"])
        self.assertIsNone(row["fan_rpm"])

    def test_hardware_metrics_enabled_calls_powermetrics(self) -> None:
        collector = MetricsCollector(Path("/tmp/metrics.csv"), enable_hardware_metrics=True)

        with mock.patch(
            "benchmark.metrics._sample_powermetrics",
            return_value={
                "gpu_utilization_percent": 12.5,
                "gpu_power_mw": 345.0,
                "die_temp_c": 55.0,
                "fan_rpm": 1200.0,
            },
        ) as sample_mock:
            row = collector._sample()

        sample_mock.assert_called_once()
        self.assertEqual(row["gpu_utilization_percent"], 12.5)
        self.assertEqual(row["gpu_power_mw"], 345.0)
