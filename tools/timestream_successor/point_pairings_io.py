"""Exact saved-contribution replay, timing reconciliation and population facts."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import netCDF4
from inspect_forward_map import read_yaml,digest,require


def checked(path,sha):
    path=Path(path).resolve();require(digest(path)==sha,f'input digest differs: {path}')
    return path


def utc(t):return datetime.fromtimestamp(float(t),timezone.utc).isoformat(timespec='milliseconds')


def scheduled_native_rows(ranges,factor,phase):
    require(isinstance(factor,int) and factor>0 and isinstance(phase,int) and 0<=phase<factor,'invalid RTC lattice')
    native=np.concatenate([np.arange(lo,hi) for lo,hi in ranges])
    return native[(native>=phase)&((native-phase)%factor==0)]


def reconcile_request(path):
    path=Path(path);request=json.loads(path.read_text());identity=digest(path)
    require(request['observation']==152389 and request['subobservation']==0 and request['scan']==2,'foreign processing request')
    config=checked(path.parent/'effective-point.yaml',request['effective_config_sha256'])
    chunking=read_yaml(config)['timestream']['chunking']
    telescope=checked(Path(request['telescope']['path']),request['telescope']['sha256'])
    with netCDF4.Dataset(telescope) as nc:
        obs={k:int(nc['Header.Dcs.'+k][...]) for k in ['ObsNum','SubObsNum','ScanNum']}
        require(list(obs.values())==[152389,0,2],'foreign telescope scope')
        scanpos=np.asarray(nc['Data.TelescopeBackend.ScanPos'][:]);starts=np.r_[0,np.flatnonzero(np.diff(scanpos)!=0)+1];ends=np.r_[starts[1:],len(scanpos)]
        telescope_facts=dict(headers=obs,requested_seconds=float(nc['Header.Dcs.RequestedTime'][...]),
            integration_seconds=float(nc['Header.Dcs.IntegrationTime'][...]),
            scanpos_runs=[dict(value=int(scanpos[i]),first_row=int(i),end_row_exclusive=int(j)) for i,j in zip(starts,ends)])
    return dict(path=str(path.resolve()),sha256=identity,processing_generation='native-processing-request:sha256:'+identity,
        effective_config=dict(path=str(config),sha256=digest(config),chunking=chunking),
        telescope=dict(path=str(telescope),sha256=digest(telescope),**telescope_facts),
        supplied_networks=[i['network'] for i in request['timing_inputs']],
        identity_resolution='raw ScanNum=2 is the observation acquisition identity; processing scan IDs name forced duration chunks, not separately labeled telescope scans',
        four_scan_reference='not identified in these bound inputs; no four-of-six selection or physical-cycle relabeling is made')


def replay_scans(root,plan,full,regions,reconciliation):
    """Caller must first run load_parent for complete MAP/PTC/VAL verification."""
    root=Path(root);operator=np.memmap(root/'operator-pixel.i32',dtype='<i4',mode='r')
    numerators={};counts={};networks={};members=[];missing=[];timing=[];artifact_hashes={};chunk_ids=None
    for q in plan['ordered_column_domain']:
        pub=read_yaml(root/q['publication_file']);segment=pub['source_segment'];network=int(pub['network']);scan=int(segment['scan'])
        source=Path(pub['source_PTC_root']);native=source.parent.parent
        if network not in networks:
            processing=read_yaml(native/'processing-scans.yaml');receipt=read_yaml(native/'receipt.yaml')
            require(processing['generation']==pub['processing']==reconciliation['processing_generation'],'processing generation differs')
            checked(native/'native-time.f64',receipt['native_time_sha256'])
            t=np.fromfile(native/'native-time.f64',dtype='<f8');require(np.isfinite(t).all() and np.all(np.diff(t)>0),'invalid native time')
            ids=[int(s['scan']) for s in processing['scans']]
            require(len(set(ids))==len(ids) and ids==sorted(ids),'processing IDs not unique chronological order')
            if chunk_ids is None:chunk_ids=ids
            require(ids==chunk_ids,'network processing scan identity mismatch')
            cal=read_yaml(Path(pub['source_CAL_root'])/'receipt.yaml');cols=segment['output_detector_indices']
            require(len(pub['detector_occurrences'])==len(cols),'detector occurrence inventory mismatch')
            networks[network]=dict(output_detector_indices=cols,occurrences=pub['detector_occurrences'],
                channels=[cal['detectors'][d]['channel'] for d in cols],processing=processing,time=t)
            for s in processing['scans']:
                ranges=s['science_native'];rows=np.concatenate([np.arange(lo,hi) for lo,hi in ranges])
                require(len(rows)>0 and np.all(np.diff(rows)>0),'empty or overlapping native science rows')
                timing.append(dict(network=network,scan=s['scan'],science_slots=s['science_slots'],science_native_half_open=ranges,
                    first_sample_unix_seconds=float(t[rows[0]]),last_sample_unix_seconds=float(t[rows[-1]]),
                    first_sample_UTC=utc(t[rows[0]]),last_sample_UTC=utc(t[rows[-1]]),science_sample_count=len(rows),
                    absolute_epoch_uncertainty=processing['absolute_epoch_uncertainty'],
                    native_outside_processing=processing['native_outside_processing']))
            ptc=read_yaml(source/'receipt.yaml')
            for s in ptc['segments']:
                if not s.get('converged',False):missing.append(dict(network=network,scan=s['scan'],cause=s.get('stopping_reason','PTC unavailable'),scope='whole segment unavailable in parent',record=s))
            for name in ['processing-scans.yaml','receipt.yaml','native-time.f64']:artifact_hashes[str(native/name)]=digest(native/name)
        nw=networks[network]
        require(segment['output_detector_indices']==nw['output_detector_indices'] and pub['detector_occurrences']==nw['occurrences'],'detector occurrence order changes across chunks')
        stem=next(n[:-len('-cleaned.f64')] for n in segment['files'] if n.endswith('-cleaned.f64'))
        shape=(segment['scheduled_times'],segment['output_detectors'])
        cause=np.fromfile(source/(stem+'-causes.u8'),dtype='u1').reshape(shape)
        values=np.fromfile(source/(stem+'-cleaned.f64'),dtype='<f8').reshape(shape)
        slots=np.fromfile(source/(stem+'-slots.i64'),dtype='<i8')
        rows=pub['RTC_native_phase']+pub['RTC_factor']*slots
        s=next(s for s in nw['processing']['scans'] if s['scan']==scan)
        expected=scheduled_native_rows(s['science_native'],pub['RTC_factor'],pub['RTC_native_phase'])
        require(np.array_equal(rows,expected),'PTC slots do not equal declared native science interval')
        retained=cause==0;offset=q['first_input_row'];pixels=np.asarray(operator[offset:offset+q['input_rows']]);admit=pixels>=0
        require(len(pixels)==int(retained.sum()),'operator/retained population mismatch')
        n=numerators.setdefault(scan,np.zeros(full['PRIMARY'].size));c=counts.setdefault(scan,np.zeros(full['PRIMARY'].size,dtype=np.int64))
        np.add.at(n,pixels[admit],values[retained][admit]);np.add.at(c,pixels[admit],1)
        detector=np.nonzero(retained)[1];region_counts={}
        for name,roi in regions.items():
            use=admit.copy();use[admit]=roi.ravel()[pixels[admit]]
            region_counts[name]=np.bincount(detector[use],minlength=shape[1]).tolist()
        # Disjoint cause codes, not guessed upstream meanings. Retain exact codes
        # and per-detector counts; excluded cells have no serialized MAP position.
        histogram={str(int(k)):np.sum(cause==k,axis=0).tolist() for k in np.unique(cause)}
        members.append(dict(network=network,scan=scan,publication=pub['identity'],PTC_root=str(source),
            RTC_factor=pub['RTC_factor'],RTC_native_phase=pub['RTC_native_phase'],
            first_RTC_sample_UTC=utc(nw['time'][rows[0]]),last_RTC_sample_UTC=utc(nw['time'][rows[-1]]),
            source_segment_files=segment['files'],rank=pub['rank'],scheduled_times=shape[0],output_detectors=shape[1],
            retained_count=int(retained.sum()),mapped_count=int(admit.sum()),
            cause_code_counts_per_output_detector=histogram,
            map_rejected_per_output_detector=np.bincount(detector[~admit],minlength=shape[1]).tolist(),region_counts=region_counts))
    require(chunk_ids is not None and set(numerators)==set(chunk_ids),'no admitted contributions for a declared scan')
    n=np.stack([numerators[s].reshape(full['PRIMARY'].shape) for s in chunk_ids]);q=np.stack([counts[s].reshape(full['PRIMARY'].shape) for s in chunk_ids])
    require(np.array_equal(q.sum(axis=0),full['HITS']) and np.array_equal(full['HITS'],full['NORMALIZATION']),'nonuniform coefficients or lost contributions')
    err=float(np.max(abs(n.sum(axis=0)-full['NUMERATOR'])));require(err<=1e-10*max(1.,np.max(abs(full['NUMERATOR']))),'numerator replay mismatch')
    inventory={str(k):{name:v for name,v in n.items() if name not in ['time','processing']} for k,n in networks.items()}
    return n,q,chunk_ids,dict(timing=timing,detector_inventory=inventory,members=members,missing_segments=missing,
        timing_artifacts_sha256=artifact_hashes,counts_exact=True,maximum_numerator_roundoff_error=err,
        exclusion_scope='PTC cause histograms apply to full segment/detector domains; excluded cells have no MAP positions and are not assigned fabricated source-region counts; admitted local populations use original operator pixels')
