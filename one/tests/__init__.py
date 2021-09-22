"""Tests for ONE-api"""
import os
import json
from pathlib import Path

"""int: Flag for skipping tests that require an http connection"""
OFFLINE_ONLY = int(os.getenv('OFFLINE_ONLY', '0'))


def _get_test_db():
    """Load test database credentials for testing ONE api

    Allows users to test ONE using their own Alyx database.  The tests use two databases: the
    first for tests requiring POST requests; the second for tests that do not affect the database.
    """
    default_fixture = str(Path(__file__).parent.joinpath('fixtures', 'test_dbs.json'))
    db_json = os.getenv('TEST_DB_CONFIG', default_fixture)
    with open(db_json, 'r') as f:
        dbs = json.load(f)
    if not isinstance(dbs, list):
        dbs = [dbs]
    return [dbs[i] if len(dbs) >= i else None for i in range(2)]  # Ensure length == 2


TEST_DB_1, TEST_DB_2 = _get_test_db()
