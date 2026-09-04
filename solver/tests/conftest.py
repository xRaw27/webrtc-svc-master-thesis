"""Pytest configuration: hypothesis test profiles.

Profiles per PLAN.md global decisions: ``fast`` (default) keeps example
counts small so CI finishes quickly; ``full`` (enabled with the environment
variable ``SFU_TESTS=full``) runs many more examples.
"""

import os

from hypothesis import settings

settings.register_profile("fast", max_examples=25, deadline=None)
settings.register_profile("full", max_examples=300, deadline=None)
settings.load_profile("full" if os.environ.get("SFU_TESTS") == "full" else "fast")
