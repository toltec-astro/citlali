"""Application sequencing with real requests/publication and bounded fake stages."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

import configured_point_cli as cli
from test_configured_point_config import config  # shared real merged-request fixture
from test_configured_point_products import native_map


@pytest.fixture
def application(tmp_path, config, monkeypatch):
    launch = tmp_path / 'project'
    launch.mkdir()
    config['runtime']['output_dir'] = str(launch / 'reduced')
    config['successor']['products']['fit_diagnostics'] = False
    config['successor']['cleaning']['ranks'] = dict(a1100=3, a1400=4, a2000=6)
    config['successor']['map'].update(pixel_size_arcsec=1.25, coverage_cut=.25)
    path = launch / 'merged.yaml'
    binary = launch / 'native'
    binary.write_text('owned fake stage executable identity\n')
    state = SimpleNamespace(launch=launch, config=config, path=path, binary=binary,
                            effective=[], calls=[], events=[], omit_array=None)
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **kw: 'citlali synthetic-test\n')

    def prepare(effective, destination, *, launch_dir, rtc_profiles):
        assert launch_dir == launch and all(p.is_file() for p in rtc_profiles.values())
        destination.mkdir()
        state.effective.append(deepcopy(effective))
        effective_path = destination / 'effective.yaml'
        effective_path.write_text(yaml.safe_dump(effective))
        request = destination / 'observation.json'
        request.write_text(json.dumps(dict(observation=152389, networks=[0, 7, 11])))
        for nw in (0, 7, 11):
            (destination / f'network{nw}-input.json').write_text(json.dumps(dict(network=nw)))
        # This input suffix must never be swept up by transport cleanup.
        (destination / 'input.f64').write_bytes(b'preserve prepared input')
        state.events.append('prepared')
        return dict(observation=152389, networks=[0, 7, 11], arrays=['a1100', 'a1400', 'a2000'],
                    effective_config=effective_path, observation_request=request,
                    bindings=[cli.bind(effective_path)], native_cells=6)

    def run(argv, *, env, stdout, stderr):
        development = yaml.safe_load(Path(argv[-1]).read_text())
        request = json.loads(Path(development['input']['path']).read_text())
        state.calls.append(dict(development=development, request=request,
                                log_level=argv[2], environment=env))
        output = Path(development['output'])
        output.mkdir()
        terminal = development['terminal']
        if terminal == 'ptc':
            for nw in request['networks']:
                root = output / f'network{nw}'
                receipt = root / 'donor-continuity/ptc/receipt.yaml'
                receipt.parent.mkdir(parents=True)
                receipt.write_text(yaml.safe_dump(dict(failed_fits=0, available=True,
                    segments=[dict(converged=True, application_failed_times=0)])))
                (root / 'receipt.yaml').write_text(yaml.safe_dump(dict(network=nw)))
                (root / 'cleaned.f64').write_bytes(b'generated upstream transport')
                (root / 'retained.dat').write_bytes(b'not a generated transport suffix')
        else:
            assert terminal == 'map'
            # MAP is still a consumer of PTC transport at this point.
            assert len(list((output.parents[1] / 'upstream/products').rglob('*.f64'))) == 3
            for array in ('a1100', 'a1400', 'a2000'):
                if array != state.omit_array:
                    native_map(output / array, array)
                    (output / array / 'operator-pixel.i32').write_bytes(b'generated operator')
        state.events.append(terminal)
        stdout.write(f'{terminal} completed fixture stage\n')
        return SimpleNamespace(returncode=0)

    original_publish = cli.publish_maps

    def publish(map_products, destination, *args, **kwargs):
        work = map_products.parents[1]
        assert list((work / 'upstream/products').rglob('*.f64'))
        assert list(map_products.rglob('*.i32'))
        result = original_publish(map_products, destination, *args, **kwargs)
        state.events.append('published')
        return result

    original_release = cli.release_transport

    def release(work):
        assert state.events[-1] == 'published'
        state.events.append('released')
        return original_release(work)

    monkeypatch.setattr(cli, 'prepare', prepare)
    monkeypatch.setattr(cli.subprocess, 'run', run)
    monkeypatch.setattr(cli, 'publish_maps', publish)
    monkeypatch.setattr(cli, 'release_transport', release)

    def execute():
        path.write_text(yaml.safe_dump(config))
        return cli.execute(path, binary, launch_dir=launch)

    state.execute = execute
    return state


def test_preserves_existing_reductions_and_releases_only_consumed_transport(application):
    app = application
    old = app.launch / 'reduced/redu03/152389/raw'
    old.mkdir(parents=True)
    reference = old / 'reference.f64'
    reference.write_bytes(b'preserved historical reduction')
    output = app.execute()
    assert output.name == 'redu04'
    assert reference.read_bytes() == b'preserved historical reduction'
    assert app.events == ['prepared', 'ptc', 'map', 'published', 'released']
    assert len(list((output / '152389/raw').glob('toltec_*.fits'))) == 3
    work = output / '.point-work'
    assert not list((work / 'upstream/products').rglob('*.f64'))
    assert not list((work / 'map/products').rglob('*.i32'))
    assert (work / 'inputs/input.f64').read_bytes() == b'preserve prepared input'
    assert len(list((work / 'upstream/products').rglob('retained.dat'))) == 3
    assert len(list((work / 'upstream/products').rglob('receipt.yaml'))) == 6
    assert len(list((work / 'map/products').rglob('map.fits'))) == 3
    record = json.loads((output / 'RUN.json').read_text())
    assert record['status'] == 'completed'
    assert record['transport_retention']['files'] == 6
    assert record['calibration_eligible'] is False
    assert app.execute().name == 'redu05'
    assert reference.read_bytes() == b'preserved historical reduction'


def test_invalid_request_rejected_before_output_or_executable_invocation(application, monkeypatch):
    app = application
    app.config['successor']['feedback']['enabled'] = True
    monkeypatch.setattr(cli.subprocess, 'check_output',
                        lambda *a, **kw: pytest.fail('executable queried for invalid request'))
    with pytest.raises(ValueError, match='feedback'):
        app.execute()
    assert not (app.launch / 'reduced').exists()
    assert not app.calls and not app.effective


def test_missing_required_array_never_publishes_observation_or_cleans_evidence(application):
    app = application
    app.omit_array = 'a1400'
    with pytest.raises(FileNotFoundError):
        app.execute()
    reduction = app.launch / 'reduced/redu00'
    assert not (reduction / '152389').exists()
    assert 'released' not in app.events
    assert len(list((reduction / '.point-work/upstream/products').rglob('*.f64'))) == 3
    assert json.loads((reduction / 'RUN.json').read_text())['status'] == 'failed'


def test_late_publication_failure_keeps_partial_products_hidden(application, monkeypatch):
    app = application

    def partial_then_fail(map_products, destination, *args, **kwargs):
        destination.mkdir(parents=True)
        (destination / 'partial-product.fits').write_bytes(b'incomplete publication')
        raise OSError('required diagnostic publication failed')

    monkeypatch.setattr(cli, 'publish_maps', partial_then_fail)
    with pytest.raises(OSError, match='required diagnostic publication failed'):
        app.execute()
    reduction = app.launch / 'reduced/redu00'
    assert not (reduction / '152389').exists()
    assert (reduction / '.point-work/publish/raw/partial-product.fits').exists()
    assert 'released' not in app.events
    assert json.loads((reduction / 'RUN.json').read_text())['status'] == 'failed'


def test_quiet_and_verbose_keep_numerical_requests_and_per_array_ranks(application):
    app = application
    app.config['runtime']['verbose'] = False
    quiet = app.execute()
    app.config['runtime']['verbose'] = True
    verbose = app.execute()
    assert [c['log_level'] for c in app.calls] == ['warning', 'warning', 'debug', 'debug']
    first, second = deepcopy(app.effective)
    assert first['runtime'].pop('verbose') is False
    assert second['runtime'].pop('verbose') is True
    assert first == second
    ranks = first['timestream']['processed_time_chunk']['clean']['standard_pca']['n_eig_to_cut']
    assert ranks == dict(a1100=[3], a1400=[4], a2000=[6])
    for ptc in (app.calls[0], app.calls[2]):
        assert ptc['development']['ptc'] == dict(method='observed-als')
        assert 'rank' not in ptc['development']['ptc']
        assert ptc['development']['network_workers'] == 6
        assert all(ptc['environment'][name] == '1' for name in
                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'))
    for call in (app.calls[1], app.calls[3]):
        assert call['request']['pixel_size_arcsec'] == 1.25
        assert call['request']['coverage_cut'] == .25
        assert len(call['request']['networks']) == 3
        assert 'ptc' not in call['development']
    for array in ('a1100', 'a1400', 'a2000'):
        name = f'toltec_point_152389_{array}_successor.fits'
        with fits.open(quiet / '152389/raw' / name) as a, fits.open(verbose / '152389/raw' / name) as b:
            assert all(np.array_equal(x.data, y.data, equal_nan=True) for x, y in zip(a, b))


def test_explicit_retention_preserves_transport(application):
    app = application
    app.config['successor']['products']['retain_intermediates'] = True
    output = app.execute()
    assert 'released' not in app.events
    assert len(list((output / '.point-work/upstream/products').rglob('*.f64'))) == 3
    assert len(list((output / '.point-work/map/products').rglob('*.i32'))) == 3
    assert json.loads((output / 'RUN.json').read_text())['transport_retention']['replay_payloads_retained']


def test_cleanup_rejects_symlink_without_touching_external_source(tmp_path):
    original = tmp_path / 'original.f64'
    original.write_bytes(b'irreplaceable input')
    work = tmp_path / 'work'
    root = work / 'upstream/products'
    root.mkdir(parents=True)
    link = root / 'foreign.f64'
    link.symlink_to(original)
    with pytest.raises(RuntimeError, match='unsafe generated transport'):
        cli.release_transport(work)
    assert link.is_symlink() and original.read_bytes() == b'irreplaceable input'


@pytest.mark.parametrize('return_code,line,fails', [
    (0, '[warning] expected fixture warning', False),
    (7, 'fixture failure without structured log', True),
    (0, '[ERROR] unexpected error despite zero exit', True),
    (0, '[critical] unexpected critical despite zero exit', True),
])
def test_native_subprocess_exit_and_error_level_output(tmp_path, return_code, line, fails, capsys):
    binary = tmp_path / 'fake-citlali'
    binary.write_text(f'#!{sys.executable}\n'
                      f'import sys\nprint({line!r})\nsys.exit({return_code})\n')
    binary.chmod(0o755)
    request = tmp_path / 'input.json'
    request.write_text('{}\n')
    destination = tmp_path / 'stage'
    if fails:
        with pytest.raises(RuntimeError, match='ptc failed'):
            cli.native_call(binary, request, destination, terminal='ptc', method='observed-als')
    else:
        cli.native_call(binary, request, destination, terminal='ptc', method='observed-als')
        assert line in capsys.readouterr().out
    record = json.loads((destination / 'stage.json').read_text())
    assert record['return_code'] == return_code
    assert bool(record['unexpected_error_lines']) == ('[ERROR]' in line or '[critical]' in line)
    assert line in (destination / 'native.log').read_text()
