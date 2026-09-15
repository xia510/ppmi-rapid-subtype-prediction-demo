import sys
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from ui.dashboard import FEATURE_NAMES, build_patient_payload, format_api_error


class DashboardHelperTests(unittest.TestCase):
    def test_build_patient_payload_preserves_feature_order_and_numeric_values(self):
        raw_values = {feature: str(index + 1) for index, feature in enumerate(FEATURE_NAMES)}

        payload = build_patient_payload("demo-ui-1", raw_values)

        self.assertEqual(payload["patient_id"], "demo-ui-1")
        self.assertEqual(list(payload.keys())[1:], FEATURE_NAMES)
        self.assertEqual(payload["scopa"], 1.0)
        self.assertEqual(payload["quip"], 12.0)

    def test_format_api_error_names_missing_feature_for_validation_response(self):
        detail = [{"type": "missing", "loc": ["body", "LEDD"]}]

        message = format_api_error(422, detail)

        self.assertIn("LEDD", message)
        self.assertIn("必填", message)


if __name__ == "__main__":
    unittest.main()
