import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_well_text():
    return (
        "Drilling report for Well Alpha-1. The well was drilled to a total depth of 8500 ft "
        "using 8.5 inch hole section. Mud weight was maintained at 10.5 ppg. Casing was set "
        "at 5000 ft with 9 5/8 inch surface casing. Drilling rate averaged 50 ft/hr through "
        "the Target formation. No lost circulation events were observed."
    )
