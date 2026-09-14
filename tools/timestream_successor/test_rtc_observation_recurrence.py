import unittest

from rtc_observation_recurrence import describe, retained_markers, summarize, cost, native_scan_summary


def detector(support=((0, 60_000_000),), groups=0):
    return dict(observation=1, network=2, detector=3, occurrence="occurrence-A",
                acquisition_scope=[1, 0, 1], array="a1100",
                program_metadata={"goal": "Science"}, health_review_concern=False,
                counts={"transition_groups": groups}, eligible_intervals_us=support,
                durations_us=dict(eligible_us=sum(b-a for a, b in support), direct_us=0,
                                  noise_screening_required_us=0))


def event(group, bounds):
    return dict(observation=1, network=2, detector=3, producer_group=group,
                apply_authorized=False, hard_event_accepted=False,
                coordinates=[dict(transition_us=b) for b in bounds])


class RecurrenceTests(unittest.TestCase):
    def test_native_absent_does_not_imply_scan_cost(self):
        result = native_scan_summary(dict(available=False))
        self.assertFalse(result["complete_native_scan_relation"])

    def test_native_counts_are_preserved_without_inventing_intervals(self):
        rtc = {k: 11 for k in ("run_count", "loaded_input_row_count", "selected_input_row_count",
                               "output_row_count", "exact_support_identity_count", "detector_support_count",
                               "flagged_detector_support_count", "final_short_support_count")}
        rtc["interval_authority"] = "telescope.scan_indices.inner_and_outer_intervals"
        value = dict(schema_version="citlali-native-cohort-product-provenance-v3",
                     observation_binding={"alignment_plan_digest": "exact-plan"},
                     scans=[dict(scan_index=0, chunk_index=0, observation_binding_digest="exact-binding",
                                 rtc=rtc, ptc={"group_count": 11}, unrelated_output={"do_not_copy": 1})])
        result = native_scan_summary(dict(available=True, value=value))
        self.assertTrue(result["available"])
        self.assertFalse(result["complete_native_scan_relation"])
        self.assertEqual(result["scans"][0]["rtc"]["exact_support_identity_count"], 11)
        self.assertNotIn("unrelated_output", result["scans"][0])
        rtc["native_intervals"] = [[0, 42]]
        with self.assertRaisesRegex(ValueError, "schema changed"):
            native_scan_summary(dict(available=True, value=value))

    def test_unknown_native_provenance_requires_reassessment(self):
        for value in ({}, {"available": True, "value": {"schema_version": "new-version"}}):
            with self.assertRaises(ValueError):
                native_scan_summary(value)

    def test_paired_coordinates_count_one_original_group(self):
        markers, n = retained_markers([event(5, [[(30, 40)], [(20, 35)]]),
                                       event(6, [[], []])])
        self.assertEqual(n, 2)
        self.assertEqual(markers[(1, 2, 3)], [(20, 5)])

    def test_neighbor_groups_are_not_merged(self):
        markers, _ = retained_markers([event(5, [[(20, 40)], []]),
                                       event(6, [[(25, 35)], []])])
        self.assertEqual(markers[(1, 2, 3)], [(20, 5), (25, 6)])

    def test_duplicate_group_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            retained_markers([event(5, [[], []]), event(5, [[], []])])

    def test_unexpected_event_admission_fails(self):
        e = event(5, [[], []])
        e["hard_event_accepted"] = True
        with self.assertRaisesRegex(ValueError, "admitted"):
            retained_markers([e])

    def test_rate_uses_exposure_not_elapsed_gap(self):
        r = describe(detector(((0, 30_000_000), (90_000_000, 120_000_000)), 2),
                     [(1, 5), (119_999_999, 6)])
        self.assertEqual(r["groups_per_eligible_minute"], 2)
        self.assertEqual(r["acquisition_or_eligibility_gap_us"], 60_000_000)
        self.assertEqual(r["quarter_eligible_us"], [30_000_000, 0, 0, 30_000_000])
        self.assertEqual(r["quarter_group_counts"], [1, 0, 0, 1])

    def test_quarter_boundary_belongs_to_right_quarter(self):
        r = describe(detector(groups=4), [(0, 1), (15_000_000, 2),
                                          (30_000_000, 3), (45_000_000, 4)])
        self.assertEqual(r["quarter_group_counts"], [1, 1, 1, 1])
        self.assertEqual(r["occupied_quarters"], 4)
        self.assertEqual(r["first_to_last_elapsed_fraction"], .75)

    def test_short_integer_span_has_empty_quarters(self):
        r = describe(detector(((0, 2),), 2), [(0, 1), (1, 2)])
        self.assertEqual(r["quarter_group_counts"], [1, 0, 1, 0])
        self.assertEqual(r["quarter_eligible_us"], [1, 0, 1, 0])

    def test_marker_in_gap_or_at_end_fails(self):
        for t in (30_000_000, 60_000_000, 120_000_000):
            with self.subTest(t=t), self.assertRaisesRegex(ValueError, "outside"):
                describe(detector(((0, 30_000_000), (90_000_000, 120_000_000)), 1), [(t, 5)])

    def test_zero_events_is_not_healthy_claim(self):
        d = detector()
        d["health_review_concern"] = True
        r = describe(d, [])
        self.assertTrue(r["health_review_concern"])
        self.assertEqual(r["groups_per_eligible_minute"], 0)
        self.assertIsNone(r["first_to_last_elapsed_fraction"])
        self.assertEqual(r["occupied_quarters"], 0)

    def test_excluded_occurrence_has_unavailable_rate(self):
        r = describe(detector(()), [])
        self.assertIsNone(r["groups_per_eligible_minute"])
        self.assertIsNone(r["occupied_quarters"])
        self.assertEqual(r["eligible_us"], 0)

    def test_changed_count_or_exposure_fails(self):
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            describe(detector(groups=1), [])
        d = detector()
        d["durations_us"]["eligible_us"] -= 1
        with self.assertRaisesRegex(ValueError, "support mismatch"):
            describe(d, [])

    def test_whole_observation_cost_and_unknown_scan_cost(self):
        r = describe(detector(), [])
        r["direct_us"] = 1_000_000
        c = cost([r], 120_000_000)
        self.assertEqual(c["full_observation_percent"], 50)
        self.assertEqual(c["additional_beyond_direct_us"], 59_000_000)
        self.assertIsNone(c["additional_beyond_full_scan_us"])

    def test_equal_counts_different_duration_and_spread(self):
        rows = [describe(detector(groups=3), [(0, 1), (20_000_000, 2), (40_000_000, 3)]),
                describe(detector(((0, 600_000_000),), 3), [(0, 1), (1000, 2), (2000, 3)])]
        s = summarize(rows)
        chosen = next(x for x in s["joint_sensitivity"] if x["minimum_groups"] == 3
                      and x["minimum_groups_per_minute"] == 1
                      and x["minimum_occupied_quarters"] == 3)
        self.assertEqual(chosen["detector_records"], 1)
        self.assertEqual(chosen["full_observation_us"], 60_000_000)
        self.assertAlmostEqual(chosen["full_observation_percent"], 100/11)

    def test_health_recovery_unresolved_are_not_transition_counts(self):
        d = detector()
        d["counts"].update(finite_recovery_groups=8, unresolved_groups=10)
        d["health_review_concern"] = True
        s = summarize([describe(d, [])])
        self.assertEqual(s["transition_groups"], 0)
        self.assertEqual(s["health_review"]["detector_records"], 1)
        for x in s["joint_sensitivity"]:
            self.assertEqual(x["detector_records"], 0)
        self.assertTrue(any(x["full_observation_percent"] == 100
                            for x in s["count_sensitivity"] if x["category"] == "unresolved_groups"))

    def test_threshold_cost_monotonicity(self):
        rows = []
        for n in (1, 2, 3, 5, 10, 21):
            rows.append(describe(detector(groups=n), [(int(i*59_000_000/n), i) for i in range(n)]))
        s = summarize(rows)
        for a in s["joint_sensitivity"]:
            for b in s["joint_sensitivity"]:
                knobs = ("minimum_groups", "minimum_groups_per_minute", "minimum_occupied_quarters")
                if all(a[k] <= b[k] for k in knobs):
                    self.assertGreaterEqual(a["full_observation_us"], b["full_observation_us"])


if __name__ == "__main__":
    unittest.main()
