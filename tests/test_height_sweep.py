import unittest

from height_sweep import summarize_height_sweep


class HeightSweepTests(unittest.TestCase):
    def test_summarizes_h1_to_h5_without_deciding_uniformity(self) -> None:
        trials = [
            {"release_level": "H3", "coefficient_of_restitution": 0.68, "uncertainty": 0.02},
            {"release_level": "H1", "coefficient_of_restitution": 0.70, "uncertainty": 0.01},
            {"release_level": "H5", "coefficient_of_restitution": 0.69, "uncertainty": 0.02},
            {"release_level": "H2", "coefficient_of_restitution": 0.72, "uncertainty": 0.01},
            {"release_level": "H4", "coefficient_of_restitution": 0.71, "uncertainty": 0.01},
        ]

        summary = summarize_height_sweep(trials)

        self.assertEqual(
            [item["release_level"] for item in summary["trials"]],
            ["H1", "H2", "H3", "H4", "H5"],
        )
        self.assertAlmostEqual(summary["mean"], 0.70, places=9)
        self.assertAlmostEqual(summary["variance"], 0.00025, places=9)
        self.assertAlmostEqual(summary["standard_deviation"], 0.0158113883, places=9)
        self.assertAlmostEqual(summary["range"], 0.04, places=9)
        self.assertAlmostEqual(summary["trend_slope_per_level"], -0.003, places=9)
        self.assertEqual(summary["monotonic_trend"], "none")
        self.assertIsNone(summary["uniformity_decision"])

    def test_zero_mean_reports_undefined_coefficient_of_variation(self) -> None:
        trials = [
            {"release_level": f"H{index}", "coefficient_of_restitution": 0.0}
            for index in range(1, 6)
        ]

        summary = summarize_height_sweep(trials)

        self.assertIsNone(summary["coefficient_of_variation"])

    def test_requires_one_result_for_each_release_level(self) -> None:
        trials = [
            {"release_level": "H1", "coefficient_of_restitution": 0.7},
            {"release_level": "H2", "coefficient_of_restitution": 0.7},
        ]

        with self.assertRaisesRegex(ValueError, "H1–H5"):
            summarize_height_sweep(trials)

    def test_rejects_trial_excluded_from_standard_statistics(self) -> None:
        trials = [
            {
                "release_level": f"H{index}",
                "coefficient_of_restitution": 0.7,
                "standard_statistics_eligible": index != 3,
            }
            for index in range(1, 6)
        ]

        with self.assertRaisesRegex(ValueError, "H3.*标准统计"):
            summarize_height_sweep(trials)


if __name__ == "__main__":
    unittest.main()
