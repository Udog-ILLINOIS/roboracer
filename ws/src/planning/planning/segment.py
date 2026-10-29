"""Overtaking segment: runs tum_segment.py in the TUM venv and reads back its answer.

Request and response are JSON files (fields: tum_segment.py's docstring). Segment rows use the raceline
columns (raceline.HEADER) but are open: s starts at 0 at the car; write them with raceline.write_raceline.
"""
import json
import os
import tempfile

from planning.raceline import TUM_PYTHON, OptimizationError, run_in_venv

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tum_segment.py')


def run_segment(request, python=TUM_PYTHON):
    """Plan one segment; returns (response dict, wall time s). Raises OptimizationError if the driver fails
    (an infeasible side is not a failure: it is reported in the response)."""
    with tempfile.TemporaryDirectory(prefix='segment_') as work:
        req, resp = os.path.join(work, 'request.json'), os.path.join(work, 'response.json')
        with open(req, 'w') as f:
            json.dump(request, f)
        try:
            _, dt = run_in_venv(DRIVER, [req, resp], resp, python=python)
        except OptimizationError as e:
            raise OptimizationError(f'segment {e}', summary=f'segment {e.summary}', ipopt_exit=e.ipopt_exit)
        try:
            with open(resp) as f:
                return json.load(f), dt
        except ValueError as e:
            raise OptimizationError(f'segment response is malformed: {e}')
