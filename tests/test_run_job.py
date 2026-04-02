"""Tests for run_job config validation."""

import unittest

from run_job import _validate_config


class RunJobValidationTests(unittest.TestCase):
    def test_validate_config_accepts_positive_integer_min_leads(self) -> None:
        config = {"client_id": "example_client", "min_leads": 3}

        _validate_config(config, "example_client")

    def test_validate_config_rejects_missing_min_leads(self) -> None:
        config = {"client_id": "example_client"}

        with self.assertRaisesRegex(ValueError, "min_leads"):
            _validate_config(config, "example_client")

    def test_validate_config_rejects_non_integer_min_leads(self) -> None:
        config = {"client_id": "example_client", "min_leads": "3"}

        with self.assertRaisesRegex(ValueError, "min_leads"):
            _validate_config(config, "example_client")

    def test_validate_config_rejects_zero_min_leads(self) -> None:
        config = {"client_id": "example_client", "min_leads": 0}

        with self.assertRaisesRegex(ValueError, "min_leads"):
            _validate_config(config, "example_client")


if __name__ == "__main__":
    unittest.main()
