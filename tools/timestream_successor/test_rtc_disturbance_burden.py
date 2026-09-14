"""Analytical exposure fixtures, not a second implementation of the detector."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset

import rtc_disturbance_burden as burden


def axis(begin=(0, 10, 20, 30), end=(10, 20, 30, 40), runs=((0, 4),)):
    first, last = np.array(begin), np.array(end)
    return burden.Axis((first + last) // 2, first, last, list(runs),
                       first / burden.US, last / burden.US, {})


class IntervalTests(unittest.TestCase):
    def test_overlapping_coordinates_events_and_guards_count_once(self):
        x, r = [(10, 30), (20, 40)], [(25, 50)]
        pair = burden.union(x + r)
        self.assertEqual(pair, [(10, 50)])
        self.assertEqual(burden.measure(pair), 40)
        self.assertEqual(burden.measure(burden.intersect(x, r)), 15)
        self.assertEqual(burden.subtract([(5, 55)], pair), [(5, 10), (50, 55)])

    def test_gaps_and_initial_invalidity_remove_exposure_not_failed_fits(self):
        acquired = [(0, 40), (60, 100)]
        eligible = burden.subtract(acquired, [(10, 20), (70, 90)])
        self.assertEqual(burden.measure(eligible), 50)
        marked = burden.intersect(eligible, [(15, 75)])
        self.assertEqual(marked, [(20, 40), (60, 70)])
        self.assertEqual(burden.measure(eligible), 50)

    def test_acquisition_gap_is_not_filled_by_row_extent(self):
        time = axis((0, 10, 100, 110), (10, 20, 110, 120), ((0, 2), (2, 4)))
        self.assertEqual(time.intervals((0, 4)), [(0, 20), (100, 120)])
        self.assertEqual(burden.measure(time.intervals((0, 4))), 40)

    def test_unequal_cadence_uses_duration_not_sample_counts(self):
        fast, slow = axis(), axis((0, 20, 40, 60), (20, 40, 60, 80))
        self.assertEqual(burden.measure(fast.intervals((0, 4))), 40)
        self.assertEqual(burden.measure(slow.intervals((0, 4))), 80)

    def test_transient_holes_do_not_create_level_shift_segments(self):
        result = burden.qualify([(0, 100)], [(45, 55)], [(10, 20)], 30)
        self.assertEqual([r["remaining_us"] for r in result["segments"]], [35, 45])
        self.assertEqual(result["all_us"], 80)
        self.assertEqual(result["longest_us"], 45)
        stricter = burden.qualify([(0, 100)], [(45, 55)], [(10, 20)], 40)
        self.assertEqual(stricter["all_us"], 45)
        self.assertEqual(burden.qualify([(0, 100)], [(45, 55)], [(10, 20)], 30, 5)["all_us"], 70)

    def test_declared_scan_boundaries_and_ordering_analytic_only(self):
        scans = [(0, 100), (100, 200)]
        shifts = [(95, 105)]  # Every intersected scan would be affected.
        costs = [burden.qualify([scan], shifts, [], 20) for scan in scans]
        self.assertEqual(sum(c["all_us"] for c in costs), 190)
        self.assertEqual(sum(c["longest_us"] for c in costs), 190)
        self.assertEqual(sum(burden.measure([s]) for s in scans if burden.intersect([s], shifts)), 200)
        more = burden.qualify([scans[0]], [(40, 45), (80, 85)], [], 20)
        self.assertEqual(more["all_us"], 75)
        self.assertEqual(more["longest_us"], 40)

    def test_invalidity_boundary_prevents_segment_join(self):
        r = burden.qualify([(0, 40), (60, 100)], [], [], 50)
        self.assertEqual(r["all_us"], 0)

    def test_extra_guard_does_not_import_events_from_another_acquisition_run(self):
        result = burden.qualify([(100, 200)], [(80, 90)], [], 0, 50)
        self.assertEqual(result["all_us"], 100)

    def test_concurrency_respects_physical_time_and_pair_unions(self):
        a = [(10, 30), (20, 40)]
        b = [(20, 50)]
        result = burden.concurrency([a, b], [(0, 100)])
        self.assertEqual(result, {0: 60, 1: 20, 2: 20})
        self.assertEqual(sum(n * dt for n, dt in result.items()), 60)

    def test_empty_complete_exclusion_and_invalid_intervals(self):
        self.assertEqual(burden.subtract([(0, 10)], [(-1, 20)]), [])
        self.assertEqual(burden.measure([]), 0)
        for invalid in ([(4, 4)], [(5, 1)], [(0.5, 2)]):
            with self.assertRaises(ValueError):
                burden.union(invalid)


class EvidenceTests(unittest.TestCase):
    def test_group_refinement_limit_survives_available_coordinate_footprints(self):
        coordinates = [dict(coordinate="x", state="finite_recovery_supported", unresolved=False),
                       dict(coordinate="r", state="no_resolved_excursion", unresolved=False)]
        self.assertEqual(burden.unresolved_group_causes(dict(refinement_limited=True), coordinates),
                         ["producer_refinement_limited"])
        self.assertEqual(burden.unresolved_group_causes(dict(refinement_limited=False), coordinates), [])

    def evidence(self):
        original = dict(seeded=True, available=True, recovery_cause=3,
                        affected=[0, 4], with_offset=dict(offset=7.))
        event = dict(coordinates=[original, original.copy()])
        initial = dict(available=True, affected=[1, 2], begin=10e-6, end=20e-6)
        final = dict(diagnostic_cause=0, primary_offset=dict(offset=9.), recovery_cause=3,
                     transition=dict(available=True, rows=[2, 3], begin=20e-6, end=30e-6))
        audit = dict(refit_requested=False, coordinates=[final, final.copy()])
        return event, initial, audit

    def test_selects_only_final_initial_or_remeasured_transition(self):
        event, initial, audit = self.evidence()
        row = burden.coordinate_evidence(event, initial, audit, dict(sigma_delta=2.), 0, axis())
        self.assertEqual(row["transition_us"], [(10, 20)])
        audit["refit_requested"] = True
        audit["coordinates"][0]["diagnostic_cause"] = 7
        row = burden.coordinate_evidence(event, initial, audit, dict(sigma_delta=2.), 0, axis())
        self.assertEqual(row["transition_us"], [(20, 30)])
        self.assertEqual(row["offset"], 9.)

    def test_rejected_shift_and_truncated_search_are_not_spikes_or_zero(self):
        event, initial, audit = self.evidence()
        audit["coordinates"][0]["diagnostic_cause"] = 5
        row = burden.coordinate_evidence(event, initial, audit, dict(sigma_delta=2.), 0, axis())
        self.assertTrue(row["unresolved"])
        self.assertEqual(row["finite_recovery_us"], [])
        self.assertEqual(row["transition_us"], [])

    def test_available_partner_does_not_resolve_unavailable_coordinate(self):
        event, initial, audit = self.evidence()
        audit["coordinates"][1]["diagnostic_cause"] = 1
        rows = [burden.coordinate_evidence(event, initial, audit, dict(sigma_delta=2.), c, axis()) for c in range(2)]
        self.assertFalse(rows[0]["unresolved"])
        self.assertTrue(rows[1]["unresolved"])

    def test_recovered_empty_excursion_is_explicit_not_missing_bound(self):
        event, initial, _ = self.evidence()
        event["coordinates"][0].update(recovery_cause=0, affected=[-1, -1])
        row = burden.coordinate_evidence(event, initial, None, dict(sigma_delta=None), 0, axis())
        self.assertEqual(row["state"], "no_resolved_excursion")
        self.assertFalse(row["unresolved"])

    def test_reassessed_recovery_cannot_borrow_original_support(self):
        event, initial, audit = self.evidence()
        event["coordinates"][0].update(recovery_cause=0, affected=[1, 2])
        audit["refit_requested"] = True
        audit["coordinates"][0].update(diagnostic_cause=4, recovery_cause=0)
        row = burden.coordinate_evidence(event, initial, audit, dict(sigma_delta=2.), 0, axis())
        self.assertEqual(row["state"], "reassessed_recovery_extent_unavailable")
        self.assertTrue(row["unresolved"])
        self.assertEqual(row["finite_recovery_us"], [])

    def test_validity_requires_complete_original_admitted_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            block = dict(detector=0, first=0, end=4,
                         coordinates=[dict(scale_cause=0, scale=1., admitted_differences=3)] * 2)
            file = folder / "health-blocks.jsonl"
            file.write_text(json.dumps(block) + "\n")
            meta = {0: dict(tune_valid=True, pair_screening_excluded_rows=0)}
            self.assertEqual(burden.verify_validity(folder, meta, axis()), {})
            block["coordinates"] = [dict(scale_cause=0, scale=1., admitted_differences=2)] * 2
            file.write_text(json.dumps(block) + "\n")
            with self.assertRaisesRegex(ValueError, "partial validity"):
                burden.verify_validity(folder, meta, axis())

    def test_duplicate_noise_coverage_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            block = dict(detector=0, first=0, end=4,
                         coordinates=[dict(scale_cause=0, scale=1., admitted_differences=3)] * 2)
            (folder / "health-blocks.jsonl").write_text((json.dumps(block) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "partition"):
                burden.verify_validity(folder, {0: dict(tune_valid=True, pair_screening_excluded_rows=0)}, axis())

    def test_source_bound_extent_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "original timing"):
            axis().bound(dict(available=True, affected=[1, 2], begin=100., end=200.))

    def test_native_clock_rounding_does_not_create_nanosecond_holes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.nc"
            with Dataset(path, "w") as f:
                f.createDimension("row", 4); f.createDimension("column", 6)
                ts = f.createVariable("Data.Toltec.Ts", "i4", ("row", "column"))
                ts[:] = [[1700000000, 10, v, i, 0, 0] for i, v in enumerate([100, 2097252, 4194405, 6291556])]
                for name, value in dict(FpgaFreq=256000000, AccumLen=2097152, SampleFreq=122.0703125,
                                        ObsNum=1, SubObsNum=0, ScanNum=2).items():
                    f.createVariable("Header.Toltec." + name, "f8")[...] = value
            result = burden.read_axis(path, dict(observation=1, subobservation=0, scan=2), dict(rows=4, physical_runs=1))
            self.assertEqual(result.intervals((0, 4)), [(-4096, 28672)])
            self.assertEqual(result.runs, [(0, 4)])
            self.assertLess(result.metadata["maximum_endpoint_rounding_us"], .01)

    def test_gzip_replay_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{i}.gz" for i in range(2)]
            for path in paths:
                with burden.compressed_writer(path) as stream:
                    burden.emit(stream, dict(a=[1, 2], missing=None))
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())


if __name__ == "__main__":
    unittest.main()
