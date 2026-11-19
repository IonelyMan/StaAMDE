import re
import unittest
from unittest.mock import patch

from phase2.lightgbm.features import API_FAMILY_PATTERNS, classify_api_family, flatten_report


def reference_families(api_call):
    return [family for family, patterns in API_FAMILY_PATTERNS.items()
            if any(re.search(pattern, api_call, re.IGNORECASE) for pattern in patterns)]


class StaticFeatureTests(unittest.TestCase):
    def test_classification_matches_original_regex_and_returns_fresh_list(self):
        calls = ["", "unrelated", "SMSMANAGER.sendTextMessage", "java.net.Socket",
                 "java.lang.reflect.Method.invoke", "Class.forName:loadClass:Runtime.exec",
                 "android.accounts.AccountManager", "AccessibilityService", "MessageDigest.Cipher"]
        calls += [pattern.replace("\\.", ".") for patterns in API_FAMILY_PATTERNS.values()
                  for pattern in patterns]
        for call in calls:
            with self.subTest(call=call):
                expected = reference_families(call)
                actual = classify_api_family(call)
                self.assertEqual(actual, expected)
                actual.append("injected")
                self.assertEqual(classify_api_family(call), expected)

    def test_overlapping_families_retain_feature_values(self):
        report = {"Static_analysis": {"API calls": {
            "Class.forName:loadClass": 3, "java.net.Socket": 2,
            "SmsManager.sendTextMessage": 1,
        }}}
        result = flatten_report(report)
        import math
        self.assertEqual(result["derived_api_family:reflection"], math.log1p(3))
        self.assertEqual(result["derived_api_family:dynamic_loading"], math.log1p(3))
        self.assertEqual(result["derived_api_family:network"], math.log1p(2))
        self.assertEqual(result["derived_api_family:sms"], math.log1p(1))

    def test_complete_report_matches_original_family_classification(self):
        report = {"Static_analysis": {
            "Permissions": ["android.permission.INTERNET"],
            "API calls": {"Class.forName:loadClass": 3, "java.net.Socket": 2, "unknown": 1},
            "Opcodes": {"invoke": 2}, "Activities": {"Example": ["ACTION"]},
            "Obfuscation": {"score": 0.25},
        }}
        actual = flatten_report(report)
        with patch("phase2.lightgbm.features.classify_api_family", side_effect=reference_families):
            expected = flatten_report(report)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
