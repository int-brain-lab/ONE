import os
import json
from pathlib import Path
OFFLINE_ONLY = int(os.getenv('OFFLINE_ONLY', '0'))


def _get_test_db():
    default_fixture = str(Path(__file__).parent.joinpath('fixtures', 'test_dbs.json'))
    db_json = os.getenv('TEST_DB_CONFIG', default_fixture)
    with open(db_json, 'r') as f:
        return json.load(f)


TEST_DB_1, TEST_DB_2 = _get_test_db()
