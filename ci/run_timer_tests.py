"""Run the focused timer tests, optionally requesting precise macOS activity."""

import os
import sys

import pytest


activity = None
process_info = None
if os.environ.get("MPL_TIMER_LATENCY_CRITICAL"):
    from Foundation import (  # type: ignore[import-not-found]
        NSActivityLatencyCritical,
        NSActivityUserInitiatedAllowingIdleSystemSleep,
        NSProcessInfo,
    )

    process_info = NSProcessInfo.processInfo()
    activity = process_info.beginActivityWithOptions_reason_(
        NSActivityLatencyCritical | NSActivityUserInitiatedAllowingIdleSystemSleep,
        "Matplotlib timer consistency CI experiment",
    )

try:
    raise SystemExit(pytest.main(sys.argv[1:]))
finally:
    if activity is not None:
        process_info.endActivity_(activity)
