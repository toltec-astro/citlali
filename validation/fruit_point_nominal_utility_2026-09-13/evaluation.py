"""Unchanged repaired evaluator; injection truth is only an external scorer."""
from common import module, REPAIR
repaired = module('utility_repaired_evaluation', REPAIR/'reevaluate.py')
measure = repaired.measure
def score_truth(data, maps, measurements, truth, definition):
    return repaired.old_evaluation.score_truth(data, maps,
        [m['original'] for m in measurements], truth, definition)
