"""Run-owned conservative storage accounting; no historical campaign walks.

Callers serialize reservation updates. Native destinations are new and disjoint.
Deletion is not credited until an explicit end-of-trajectory reconciliation.
"""
import json
import shutil
from pathlib import Path
from inspect_forward_map import require
from point_additive_diagnostics import save_json


def tree_bytes(root):
    total = 0
    for p in Path(root).rglob('*'):
        require(not p.is_symlink(), 'storage accounting forbids symlinked output')
        if p.is_file():
            total += p.stat().st_size
    return total


class RunStorage:
    def __init__(self, root, budget):
        self.root = Path(root)
        self.budget = budget
        self.path = self.root / 'storage-ledger.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            scope=Path(budget.get('campaign_root',self.root))
            historical=budget.get('campaign_retained_bytes')
            if historical is None:
                historical=tree_bytes(scope)-tree_bytes(self.root) if scope!=self.root else 0
            self.state = dict(historical_bytes=historical,charged_bytes=tree_bytes(self.root),
                              native_written_bytes=0, completed_stages=0)
        self.historical = self.state['historical_bytes']
        # Bounds untallied JSON/log/request/fit output between native stages.
        self.metadata = budget.get('metadata_reserve_bytes', 100_000_000)

    def reserve(self, allowance, other_reserved=0):
        require(allowance > 0, 'positive storage reservation required')
        require(self.historical + self.state['charged_bytes'] + self.metadata +
                other_reserved + allowance <= self.budget['peak_ceiling_bytes'],
                'reserved campaign storage ceiling')
        require(shutil.disk_usage(self.root).free > other_reserved + allowance +
                self.budget['free_space_reserve_bytes'], 'free-space reserve')
        save_json(self.path,self.state)

    def finish(self, destination, allowance):
        destination = Path(destination)
        require(destination.resolve().is_relative_to(self.root.resolve()) and
                not destination.is_symlink(), 'foreign native destination')
        # Exactly one reconciliation of this newly written stage, including failure.
        written = tree_bytes(destination)
        self.state['charged_bytes'] += written
        self.state['native_written_bytes'] += written
        self.state['completed_stages'] += 1
        self.state['last_stage_bytes'] = written
        save_json(self.path, self.state)
        require(written <= allowance, 'native stage exceeded reserved storage')
        require(self.historical + self.state['charged_bytes'] + self.metadata <=
                self.budget['peak_ceiling_bytes'], 'actual campaign storage ceiling')
        require(shutil.disk_usage(self.root).free >= self.budget['free_space_reserve_bytes'],
                'actual free-space reserve')
        return dict(**self.state,
                    metadata_reserve_bytes=self.metadata)

    def reconcile(self):
        actual = tree_bytes(self.root)
        require(actual <= self.state['charged_bytes'] + self.metadata,
                'untracked writes exceeded metadata reserve')
        require(self.historical + actual <= self.budget['peak_ceiling_bytes'],
                'reconciled campaign storage ceiling')
        self.state['reconciled_actual_bytes'] = actual
        save_json(self.path, self.state)
        return self.state
