import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List

API_FAMILY_PATTERNS = {
    "sms": [r"SmsManager", r"sendTextMessage", r"sendMultipartTextMessage"],
    "telephony": [r"TelephonyManager", r"getDeviceId", r"getSubscriberId", r"getLine1Number"],
    "location": [r"LocationManager", r"requestLocationUpdates", r"getLastKnownLocation"],
    "contacts": [r"ContactsContract"],
    "account": [r"AccountManager", r"android\.accounts"],
    "package_manager": [r"PackageManager", r"getInstalledPackages", r"getInstalledApplications"],
    "reflection": [r"java\.lang\.reflect", r"Class\.forName", r"getDeclaredMethod", r"\.invoke"],
    "dynamic_loading": [r"DexClassLoader", r"PathClassLoader", r"loadClass", r"loadLibrary"],
    "crypto": [r"javax\.crypto", r"java\.security", r"MessageDigest", r"Cipher"],
    "network": [r"java\.net", r"HttpURLConnection", r"Socket", r"okhttp", r"retrofit"],
    "file_io": [r"java\.io", r"Environment", r"openFileOutput"],
    "shell": [r"Runtime\.exec", r"ProcessBuilder"],
    "accessibility": [r"AccessibilityService", r"accessibilityservice"],
    "device_admin": [r"DeviceAdminReceiver", r"DevicePolicyManager"],
}

# Each family is a disjunction of case-insensitive patterns. Compile once per
# process and reuse classifications for API names repeated across APKs.
_API_FAMILY_REGEX = tuple(
    (family, re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE))
    for family, patterns in API_FAMILY_PATTERNS.items()
)


def add_feature(features: Dict[str, float], name: str, value: float = 1.0) -> None:
    if name:
        features[name] = features.get(name, 0.0) + float(value)


def add_count_dict(features: Dict[str, float], prefix: str, values, use_log: bool = True) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 1.0
        add_feature(features, f"{prefix}:{key}", math.log1p(numeric) if use_log else numeric)


@lru_cache(maxsize=65536)
def _classify_api_family_cached(api_call: str) -> tuple:
    return tuple(family for family, pattern in _API_FAMILY_REGEX if pattern.search(api_call))


def classify_api_family(api_call: str) -> List[str]:
    # Return a fresh list as before, so callers cannot mutate the cache.
    return list(_classify_api_family_cached(api_call))


def flatten_report(report: Dict[str, object]) -> Dict[str, float]:
    static = report.get("Static_analysis", {}) or {}
    features: Dict[str, float] = {}

    permissions = static.get("Permissions", []) or []
    for permission in permissions:
        add_feature(features, f"permission:{permission}")
    add_feature(features, "summary:permission_count", len(permissions))

    add_count_dict(features, "opcode", static.get("Opcodes", {}))
    add_count_dict(features, "api", static.get("API calls", {}))
    add_count_dict(features, "api_pkg", static.get("API packages", {}))
    add_count_dict(features, "api_family", static.get("API families", {}))
    add_count_dict(features, "cmd", static.get("System commands", {}))
    add_count_dict(features, "intent", static.get("Intents", {}))
    add_count_dict(features, "suspicious_string", static.get("Suspicious strings", {}))

    api_calls = static.get("API calls", {}) or {}
    if isinstance(api_calls, dict):
        family_counter = Counter()
        total_api_calls = 0.0
        for api_call, count in api_calls.items():
            total_api_calls += float(count)
            for family in classify_api_family(str(api_call)):
                family_counter[family] += float(count)
        for family, count in family_counter.items():
            add_feature(features, f"derived_api_family:{family}", math.log1p(count))
        add_feature(features, "summary:api_unique_count", len(api_calls))
        add_feature(features, "summary:api_total_count", math.log1p(total_api_calls))

    for component_key in ["Activities", "Services", "Receivers"]:
        components = static.get(component_key, {}) or {}
        add_feature(features, f"summary:{component_key.lower()}_count", len(components))
        if isinstance(components, dict):
            for component_name, intents in components.items():
                add_feature(features, f"component:{component_key}:{component_name}")
                if isinstance(intents, list):
                    add_feature(features, f"summary:{component_key.lower()}_intent_count", len(intents))

    obfuscation = static.get("Obfuscation", {}) or {}
    if isinstance(obfuscation, dict):
        for key, value in obfuscation.items():
            try:
                add_feature(features, f"obfuscation:{key}", float(value))
            except (TypeError, ValueError):
                continue

    return features
