"""Root conftest — sets default env vars for test runs."""

import os

os.environ.setdefault(
    "DR_LLM_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/dr_llm_test",
)
