"""Exercise the actual default executable; no route flag or alternate binary."""
import argparse
import hashlib
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class SuccessorDefaultCli(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="citlali-default-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"

    def invoke(self, config, *overlays):
        paths = []
        for index, content in enumerate((config, *overlays)):
            path = self.root / f"config-{index}.json"
            path.write_text(json.dumps(content))
            paths.append(str(path))
        return subprocess.run([CLI, *paths], text=True, capture_output=True, timeout=30)

    def request(self, contents):
        path = self.root / "input.json"
        path.write_text(json.dumps(contents))
        return dict(schema="citlali-development-v1", terminal="rtc-only",
                    input=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    output=str(self.output))

    def rejected(self, result, reason):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(reason, result.stdout + result.stderr)
        self.assertIn("Timestream Successor", result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_dump_config_describes_default_successor_and_actual_stop(self):
        result = subprocess.run([CLI, "--dump_config"], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("schema: citlali-development-v1", result.stdout)
        self.assertIn("terminal: ptc", result.stdout)
        self.assertNotIn("mapmaking:", result.stdout)

    def test_point_observation_rejects_incomplete_sequence_before_publication(self):
        health = self.root / "health-request.json"
        health.write_text("{}\n")
        output = self.root / "health/products"
        config = self.root / "session.json"
        config.write_text(json.dumps(dict(schema="citlali-development-v1", terminal="point-observation",
            input=dict(path=str(health), sha256=hashlib.sha256(health.read_bytes()).hexdigest()),
            output=str(output), network_workers=3)))
        for command, reason in [("finish\n", "finished before M0"), ("", "control stream ended"),
                                ("bad\n", "invalid POINT pass command")]:
            with self.subTest(command=command):
                read_fd, write_fd = os.pipe()
                try:
                    result = subprocess.run([CLI, str(config)], text=True, input=command,
                        capture_output=True, timeout=30, pass_fds=(write_fd,),
                        env=dict(os.environ, CITLALI_POINT_REPLY_FD=str(write_fd)))
                finally:
                    os.close(write_fd)
                    os.close(read_fd)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stdout + result.stderr)
                self.assertFalse(output.exists())

    def test_point_observation_checks_first_scientific_stage_before_cal_loading(self):
        health = self.root / "health-request.json"
        health.write_text(json.dumps(dict(schema="citlali-configured-weighted-point-v1", stage="cycle", cycle_index=0)))
        stage = self.root / "health"
        stage.mkdir()
        binding = dict(path=str(health), sha256=hashlib.sha256(health.read_bytes()).hexdigest())
        descriptor = stage / "development.yaml"
        cfg = dict(schema="citlali-development-v1", input=binding, output=str(stage/"products"),
                   terminal="map", network_workers=3)
        descriptor.write_text(json.dumps(cfg))
        launch = self.root / "session.yaml"
        launch.write_text(json.dumps(dict(cfg, terminal="point-observation")))
        read_fd, write_fd = os.pipe()
        try:
            result = subprocess.run([CLI, str(launch)], text=True,
                input=hashlib.sha256(descriptor.read_bytes()).hexdigest()+"\n",
                capture_output=True, timeout=30, pass_fds=(write_fd,),
                env=dict(os.environ, CITLALI_POINT_REPLY_FD=str(write_fd)))
        finally:
            os.close(write_fd)
            os.close(read_fd)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("POINT session stage order or frozen health differs", result.stdout+result.stderr)
        self.assertFalse((stage/"products").exists())

    def test_point_cal_session_requires_its_bound_upstream_descriptor(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        observation = inputs / "observation.json"
        observation.write_text("{}\n")
        upstream = self.root / "upstream"
        upstream.mkdir()
        output = upstream / "products"
        descriptor = upstream / "development.yaml"
        binding = dict(path=str(observation), sha256=hashlib.sha256(observation.read_bytes()).hexdigest())
        settings = dict(schema="citlali-development-v1", input=binding, output=str(output),
                        terminal="point-cal-observation", network_workers=3)
        launch = self.root / "launch.yaml"
        launch.write_text(json.dumps(settings))
        descriptor.write_text(json.dumps(dict(settings, terminal="map")))
        read_fd, write_fd = os.pipe()
        try:
            result = subprocess.run([CLI, str(launch)], text=True, timeout=30,
                input=hashlib.sha256(descriptor.read_bytes()).hexdigest()+"\n",
                capture_output=True, pass_fds=(write_fd,),
                env=dict(os.environ, CITLALI_POINT_REPLY_FD=str(write_fd)))
        finally:
            os.close(write_fd)
            os.close(read_fd)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("POINT pass descriptor route or binding differs", result.stdout+result.stderr)
        self.assertFalse(output.exists())

    def native_request_with_bad_raw_digest(self):
        raw = self.root / "raw-placeholder"
        raw.write_text("deliberately not detector data")
        return dict(schema="rtc-multidetector-experiment-v1",
                    observation=152389, network=7,
                    terminal_request="rtc-only", decision_apply={},
                    reassessment_selection=dict(intent="retain-development-candidate"),
                    effective_config=self.ptc_effective_binding(),
                    manifest=dict(path=str(raw), sha256="0" * 64),
                    raw=dict(path=str(raw), sha256="0" * 64))

    def ptc_effective_binding(self, **changes):
        cleaning = dict(enabled=True, grouping=["nw"], mask_radius_arcsec=0,
                        standard_pca=dict(enabled=True, stddev_limit=0,
                            n_eig_to_cut=dict(a1100=[3], a1400=[7], a2000=[10])))
        cleaning.update(changes)
        path = self.root / "effective.json"
        path.write_text(json.dumps(dict(timestream=dict(processed_time_chunk=dict(clean=cleaning)))))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_ptc_config_rejected_before_output_claim_and_native_ingress(self):
        for changes, reason in (
            (dict(grouping=["array"]), "successor.PTC_grouping_unavailable"),
            (dict(mask_radius_arcsec=1), "successor.PTC_selection_unavailable"),
            (dict(adaptive_selector=dict(enabled=True)), "successor.PTC_selection_unavailable"),
            (dict(tau=1), "successor.PTC_tau_selection_unavailable"),
        ):
            with self.subTest(changes=changes):
                native = self.native_request_with_bad_raw_digest()
                native["effective_config"] = self.ptc_effective_binding(**changes)
                config = self.request(native)
                config["terminal"] = "ptc"
                result = self.invoke(config)
                self.rejected(result, reason)
                self.assertNotIn("input digest mismatch", result.stdout + result.stderr)
                self.assertIn("exit_code=1", result.stdout + result.stderr)

    def test_observation_ptc_config_rejected_before_shared_preparation(self):
        effective = self.ptc_effective_binding(grouping=["array"])
        entries = []
        for nw in (0, 7):
            network = self.root / f"network{nw}.json"
            network.write_text(json.dumps(dict(network=nw, observation=152389,
                decision_apply={}, filter_plan={}, effective_config=effective)))
            entries.append(dict(network=nw, input=dict(path=str(network),
                sha256=hashlib.sha256(network.read_bytes()).hexdigest())))
        config = self.request(dict(schema="citlali-observation-networks-v1", observation=152389,
                                   subobservation=0, scan=2, networks=entries))
        config["terminal"] = "ptc"
        config["network_workers"] = 8
        self.rejected(self.invoke(config), "successor.PTC_grouping_unavailable")

    def test_rtc_only_does_not_activate_or_validate_unused_ptc(self):
        native = self.native_request_with_bad_raw_digest()
        native["effective_config"] = self.ptc_effective_binding(grouping=["array"])
        result = self.invoke(self.request(native))
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input digest mismatch", combined)
        self.assertNotIn("successor.PTC_grouping_unavailable", combined)
        self.assertFalse((self.output / "donor-continuity" / "ptc").exists())
        self.assertIn("Native setup: network=7 terminal=rtc-only", combined)
        self.assertIn("execution not yet observed", combined)
        self.assertIn("exit_code=1", combined)
        self.assertNotIn("Native result:", combined)
        setup = (self.output / "setup.yaml").read_text()
        self.assertIn("setup-only;execution-not-yet-observed", setup)
        self.assertNotIn("PTC:", setup)

    def test_existing_output_is_rejected_before_native_ingress(self):
        config = self.request(self.native_request_with_bad_raw_digest())
        self.output.mkdir()
        sentinel = self.output / "performance.jsonl"
        sentinel.write_text("preserve existing attempt")
        result = self.invoke(config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("native output already claimed", result.stdout + result.stderr)
        self.assertNotIn("input digest mismatch", result.stdout + result.stderr)
        self.assertEqual(sentinel.read_text(), "preserve existing attempt")

    def test_native_terminal_directories_exist_before_raw_ingress(self):
        config = self.request(self.native_request_with_bad_raw_digest())
        config["terminal"] = "ptc"
        result = self.invoke(config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input digest mismatch", result.stdout + result.stderr)
        for name in ("cal", "ptc"):
            self.assertTrue((self.output / "donor-continuity" / name).is_dir())
        self.assertTrue((self.output / "performance.jsonl").is_file())
        self.assertFalse((self.output / "receipt.yaml").exists())

    def test_symlink_parent_dotdot_existing_output_fails_before_ingress(self):
        work = self.root / "work"
        storage = self.root / "storage"
        work.mkdir()
        (storage / "child").mkdir(parents=True)
        (storage / "run").mkdir()
        (work / "link").symlink_to(storage / "child", target_is_directory=True)
        sentinel = storage / "run" / "performance.jsonl"
        sentinel.write_text("preserve physical attempt")
        config = self.request(self.native_request_with_bad_raw_digest())
        for suffix in ("", "/"):
            with self.subTest(trailing_separator=bool(suffix)):
                config["output"] = str(work / "link" / ".." / "run") + suffix
                result = self.invoke(config)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("native output already claimed", result.stdout + result.stderr)
                self.assertNotIn("input digest mismatch", result.stdout + result.stderr)
                self.assertEqual(sentinel.read_text(), "preserve physical attempt")
                self.assertFalse((work / "run").exists())
                self.assertFalse((storage / "run" / "donor-continuity").exists())

    def test_all_network_directories_exist_before_shared_preparation(self):
        entries = []
        effective = self.ptc_effective_binding()
        for nw in (0, 7):
            network = self.root / f"network{nw}.json"
            network.write_text(json.dumps(dict(network=nw, observation=152389,
                                               decision_apply={}, filter_plan={}, effective_config=effective)))
            entries.append(dict(network=nw, input=dict(path=str(network),
                sha256=hashlib.sha256(network.read_bytes()).hexdigest())))
        config = self.request(dict(schema="citlali-observation-networks-v1", observation=152389,
                                   subobservation=0, scan=2, networks=entries))
        config["terminal"] = "ptc"
        result = self.invoke(config)  # Deliberately missing shared scientific inputs.
        self.assertNotEqual(result.returncode, 0)
        for nw in (0, 7):
            network_output = self.output / f"network{nw}"
            for name in ("cal", "ptc"):
                self.assertTrue((network_output / "donor-continuity" / name).is_dir())
            self.assertFalse((network_output / "performance.jsonl").exists())

    def test_legacy_config_cannot_silently_reach_old_processing(self):
        self.rejected(self.invoke(dict(runtime=dict(reduction_type="science"), data_items=[])),
                      "successor.input_binding_unavailable")

    def test_downstream_request_does_not_fall_back_to_rtc_completion(self):
        config = self.request({})
        config["terminal"] = "ordinary-science"
        self.rejected(self.invoke(config), "successor.endpoint_not_implemented")

    def test_network_worker_bounds_and_single_network_misuse(self):
        config = self.request({})
        for workers in (0, -1, 33):
            config['network_workers'] = workers
            self.rejected(self.invoke(config), 'successor.network_workers_must_be_1_to_32')
        for workers in (2, 8, 12, 13, 32):
            config['network_workers'] = workers
            self.rejected(self.invoke(config), 'successor.network_workers_require_observation_request')

    def test_science_observation_budget_is_explicit_and_point_stays_bounded(self):
        config = self.request(dict(schema='citlali-observation-networks-v1',
            development_reference='configurable-point-v1', mode='POINT',
            observation=152389, subobservation=0, scan=2, networks=[]))
        config['network_workers'] = 13
        self.rejected(self.invoke(config), 'successor.network_worker_budget_outside_profile')
        config = self.request(dict(schema='citlali-observation-networks-v1',
            development_reference='configurable-science-compact-v1', mode='SCIENCE',
            observation=152389, subobservation=0, scan=2, networks=[]))
        config['network_workers'] = 32
        self.rejected(self.invoke(config), 'successor.empty_network_request')

    def test_observation_inventory_rejects_duplicate_network_before_preparation(self):
        network = self.root / 'network.json'
        network.write_text(json.dumps(dict(network=0, observation=152390, decision_apply={}, filter_plan={})))
        entry = dict(network=0, input=dict(path=str(network), sha256=hashlib.sha256(network.read_bytes()).hexdigest()))
        config = self.request(dict(schema='citlali-observation-networks-v1', observation=152390,
                                   subobservation=0, scan=2, networks=[entry, entry]))
        for workers in (1, 8, 12):
            config['network_workers'] = workers
            self.rejected(self.invoke(config), 'successor.network_order_duplicate_or_invalid')

    def test_observation_cannot_silently_share_different_input_authorities(self):
        entries = []
        for nw in (0, 7):
            network = self.root / f'network{nw}.json'
            network.write_text(json.dumps(dict(network=nw, observation=152390, decision_apply={},
                                               filter_plan={}, telescope=dict(sha256=str(nw)))))
            entries.append(dict(network=nw, input=dict(path=str(network), sha256=hashlib.sha256(network.read_bytes()).hexdigest())))
        config = self.request(dict(schema='citlali-observation-networks-v1', observation=152390,
                                   subobservation=0, scan=2, networks=entries))
        self.rejected(self.invoke(config), 'successor.shared_input_mismatch: telescope')

    def test_point_inventory_cannot_import_science_observation_network(self):
        network = self.root / 'network.json'
        network.write_text(json.dumps(dict(network=0, observation=152390, decision_apply={}, filter_plan={})))
        entry = dict(network=0, input=dict(path=str(network), sha256=hashlib.sha256(network.read_bytes()).hexdigest()))
        config = self.request(dict(schema='citlali-observation-networks-v1', observation=152389,
                                   subobservation=0, scan=2, networks=[entry]))
        self.rejected(self.invoke(config), 'successor.network_binding_incomplete')

    def test_point_scope_does_not_admit_other_observations_or_scans(self):
        for obs, sub, scan in ((152388,0,2), (152389,1,2), (152389,0,1)):
            config = self.request(dict(schema='citlali-observation-networks-v1', observation=obs,
                                       subobservation=sub, scan=scan, networks=[]))
            self.rejected(self.invoke(config), 'successor.observation_scope_unsupported')

    def test_point_inventory_requires_matching_shared_parents(self):
        entries = []
        for nw in (0, 7):
            network = self.root / f'network{nw}.json'
            network.write_text(json.dumps(dict(network=nw, observation=152389, decision_apply={},
                                               filter_plan={}, telescope=dict(sha256=str(nw)))))
            entries.append(dict(network=nw, input=dict(path=str(network), sha256=hashlib.sha256(network.read_bytes()).hexdigest())))
        config = self.request(dict(schema='citlali-observation-networks-v1', observation=152389,
                                   subobservation=0, scan=2, networks=entries))
        self.rejected(self.invoke(config), 'successor.shared_input_mismatch: telescope')

    def test_bound_request_tamper_is_rejected_before_execution(self):
        config = self.request({})
        Path(config["input"]["path"]).write_text("{\"changed\":true}")
        self.rejected(self.invoke(config), "successor.input_digest_mismatch")

    def test_unsupported_extra_settings_are_not_ignored(self):
        config = self.request({})
        config["legacy_fallback"] = True
        self.rejected(self.invoke(config), "successor.unsupported_configuration")

    def test_merged_configuration_uses_the_same_default(self):
        config = self.request({})
        self.rejected(self.invoke(config, dict(terminal="PTC")), "successor.endpoint_not_implemented")

    def test_missing_terminal_defaults_to_ptc_and_enters_the_existing_rtc_parent(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1"))
        del config["terminal"]
        self.rejected(self.invoke(config), "successor.RTC_plan_incomplete")

    def test_unknown_ptc_method_and_nonpositive_rank_fail_before_ingress(self):
        config = self.request({})
        config["terminal"] = "ptc"
        config["ptc"] = dict(method="magic")
        self.rejected(self.invoke(config), "successor.PTC_method_unknown")
        config["ptc"] = dict(rank=0)
        self.rejected(self.invoke(config), "successor.PTC_rank_must_be_positive")

    def test_saved_map_rejects_raw_or_cal_and_parallel_override(self):
        for schema in ('rtc-multidetector-experiment-v1','citlali-cal-output-v1'):
            config=self.request(dict(schema=schema));config['terminal']='map'
            self.rejected(self.invoke(config),'MAP requires an exact saved-PTC input request')
        config=self.request({});config['terminal']='map';config['network_workers']=2
        self.rejected(self.invoke(config),'successor.saved_MAP_requires_serial_execution')

    def test_weighted_saved_map_accepts_worker_budget_but_rejects_ptc_override(self):
        # No scientific files: the weighted owner's stage rejection proves
        # dispatch with the worker budget while failing before any loading.
        config=self.request(dict(schema='citlali-configured-weighted-point-v1', stage='invalid'))
        config['terminal']='map';config['network_workers']=3
        result=self.invoke(config)
        self.rejected(result,'weighted POINT unknown stage')
        config['ptc']=dict(rank=3)
        self.rejected(self.invoke(config),'successor.PTC_request_invalid')

    def test_actual_rtc_boundary_refuses_missing_frozen_plan(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1"))
        self.rejected(self.invoke(config), "successor.RTC_plan_incomplete")

    def test_experimental_injection_not_enabled_by_default_activation(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1",
            terminal_request="rtc-only", decision_apply={}, declared_contaminant={}))
        self.rejected(self.invoke(config), "successor.unsupported_experiment")

    def test_relative_paths_cannot_acquire_ambient_binding(self):
        config = self.request({})
        config["input"]["path"] = "input.json"
        self.rejected(self.invoke(config), "successor.path_binding")

    def test_fixed_plan_injection_trials_are_not_default_reductions(self):
        config = self.request(dict(schema="rtc-multidetector-experiment-v1",
            terminal_request="rtc-only", decision_apply={},
            fixed_plan_injections=[dict(amplitude=1.0)]))
        self.rejected(self.invoke(config), "successor.unsupported_experiment")

    def test_guarded_feedback_matrix_and_frozen_gates_before_execution(self):
        # Stop at a deliberately invalid candidate digest after the new gate;
        # no detector inputs or observation processing occur in these fixtures.
        def bound(name, value):
            p=self.root/name;p.write_text(json.dumps(value))
            return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        mo=bound('map-owner.json',{})
        parent=bound('parent.json',dict(array='a1100',observation='obs:152389_0_2',coverage_cut=.1,owner_binding='sha256:'+mo['sha256']))
        mr=bound('map.json',dict(schema='citlali-map-saved-ptc-v1',coverage_cut=.1,pixel_size_arcsec=2,owner_binding=mo))
        checks=dict(repair=dict(a1100=dict(usable=True,source_retention_check_pass=True)))
        gate=dict(policies=dict(a1100={p:dict(comparison_usable=True,threshold=4.) for p in ('CORR','DIAG')}))
        owner=dict(identity='POINT-152389-GUARDED-OWNER-2026-09-25',outside_disk_no_feedback=True,
                   direct_identical_operand_rejoin=True,gain=1,nonzero_cycles=10,source_method='POINT-EXPECTED-v1',
                   prospective=bound('prospective.json',{}),threshold=4.,
                   thresholds=bound('thresholds.json',dict(thresholds=dict(a1100=dict(CORR=4.,DIAG=4.)))))
        bad=bound('bad-candidate.json',{});bad['sha256']='0'*64
        def invoke(policy,branch='REF',cycle=1,reason='input digest mismatch:'):
            owner.update(policy=policy,source_checks=bound('checks.json',checks),launch_gate=bound('gate.json',gate))
            req=dict(schema='citlali-point-fruit-guarded-v1',method='point-152389-guarded-feedback-v1',
                     mode='zero' if cycle==0 else 'reference',array='a1100',branch=branch,cycle_index=cycle,
                     map_request=mr,parent_plan=parent,owner_binding=bound('owner.json',owner),candidate=bad)
            config=self.request(req);config['terminal']='point-fruit'
            self.rejected(self.invoke(config),reason)
        for policy in ('BASE-POINT','CORR','DIAG'):
            for branch in ('REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder'):
                with self.subTest(policy=policy,branch=branch):invoke(policy,branch)
        invoke('BASE',reason='FRUIT unknown guarded policy')
        for branch in ('D-seed-low','D-seed-high','D-threshold-5'):
            invoke('BASE-POINT',branch,reason='FRUIT guarded matrix outside bound')
        invoke('CORR',cycle=0,reason='FRUIT guarded matrix outside bound')
        invoke('CORR',cycle=11,reason='FRUIT C+D branch/index/mode outside bound')
        gate['policies']['a1100']['CORR']['comparison_usable']=False
        invoke('CORR',reason='FRUIT guarded launch gate failed')
        owner['threshold']=3.
        invoke('DIAG',reason='FRUIT guarded frozen threshold differs')
        checks['repair']['a1100']['source_retention_check_pass']=False
        invoke('BASE-POINT',reason='FRUIT guarded source checks failed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    args, remaining = parser.parse_known_args()
    CLI = args.cli
    unittest.main(argv=[__file__, *remaining])
