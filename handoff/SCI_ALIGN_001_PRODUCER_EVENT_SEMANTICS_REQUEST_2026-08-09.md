# SCI-ALIGN-001 producer event-semantics request

Date prepared: 2026-08-09

Status: **READY FOR OWNER TO SEND — no contact made by Codex**

## Human-sendable request

Subject: Request for exact TolTEC detector acquisition event semantics

Please provide either the exact source route (A) or the authoritative
specification route (B) below for this fixed corpus:

- observations 148670, 150819, and 151126;
- map identities `map:912d0ccf8b3539501f6c`,
  `map:1971b4dfddbc99932afb`, and `map:d5fec4dcd0f16b424fb6`;
- networks 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, and 12;
- frozen group `roach-t0:44cf69da97d473965ef6`.

The accepted audit at Citlali commit
`08f0a6733d1cb523ae78ccf9348ac6832b834e52` proves that each delivered
detector row preserves its `Is`/`Qs`/`Ts` association through Citlali. It does
not establish whether that row's time denotes integration start, end,
effective centroid, counter capture, packet formation, or another physical
event. This request seeks provenance and event ordering only. It does not
presume an acquisition-hardware fault or request a timing correction.

### A. Minimum exact-source response

Please provide an immutable source revision and only the files needed to trace
one completed detector sample from accumulation through the delivered NetCDF
row:

1. FPGA/gateware modules for detector accumulation or decimation, completion
   or sample-ready strobes, internal-clock and packet counters, PPS capture,
   clock-domain crossings, data/metadata latching, and packet FIFO insertion.
2. Firmware/backend code that initializes T0, handles the PPS interrupt or
   update, reads FPGA fields, and constructs the UDP payload.
3. The UDP wire-format declarations and any register map shared across FPGA,
   firmware, and host code.
4. Host receiver/parser and NetCDF-writer code that assigns packet payloads to
   `Data.Toltec.Is`, `Data.Toltec.Qs`, `Data.Toltec.Ts`, and
   `Data.Toltec.RecvTime` at row index `n`, including drop, duplicate,
   out-of-order, and rollover handling.
5. Initialization/reset code and build metadata needed to interpret
   `Header.Toltec.CompileTime`, `Header.Toltec.FpgaFreq`,
   `Header.Toltec.AccumLen`, and `Header.Toltec.SampleFreq`.
6. A map/network-to-revision table for all 33 map/network combinations above,
   including bitstream, firmware/backend, and NetCDF-writer identities. Do not
   substitute the current main branch for the deployed revisions.

A whole-repository archive is unnecessary if the supplied file set includes
all transitive definitions needed to establish capture and indexing order.

### B. Minimum authoritative-specification response

If exact source cannot be supplied, please provide a versioned, technically
approved event-level specification that covers the same six items. It must
identify the deployed build/revision for every map/network combination and
include an ordering diagram or equivalent pseudocode binding integration,
counter capture, PPS update, UDP assembly, and NetCDF row insertion. A generic
statement such as “the timestamp is the sample time” is insufficient.

### Exact questions to answer

Please answer separately for every distinct deployed revision:

1. What exact input interval contributes to `Is[n,*]` and `Qs[n,*]`? State
   whether bounds are inclusive/exclusive and whether `AccumLen=2097152` at
   `FpgaFreq=256000000 Hz` is a contiguous integration, a decimated output
   cadence, or something else.
2. Which physical event is the intended time of row `n`: integration start,
   end, effective centroid, output-valid/latch, counter snapshot, packet
   formation, or another event? Give any fixed digital-filter, pipeline, or
   transport latency in FPGA ticks and state whether it varies by network or
   revision.
3. At what event are `Is[n,*]`, `Qs[n,*]`, `ClockCount[n]`,
   `PacketCount[n]`, `PpsCount[n]`, and `PpsTime[n]` latched? Are they one
   atomic snapshot? If not, state the ordering and whether any field may refer
   to integration `n-1` or `n+1`.
4. What are the exact epoch, units, width, rollover rule, reset rule, and
   increment event for each `Data.Toltec.Ts` field:
   `ClockTime`, `PpsCount`, `ClockCount`, `PacketCount`, `PpsTime`, and
   `ClockTimeNanoSec`?
5. What does `PpsTime` represent: the internal-clock value latched on a PPS
   edge, the latest observed PPS value, or another quantity? When does it
   become visible to the sample/packet path?
6. In what order do the PPS edge, PPS ISR/update, `PpsCount` increment,
   `PpsTime` update, detector completion, and packet snapshot occur? Describe
   behavior when a PPS edge coincides with or crosses a detector sample
   boundary.
7. Does the PPS event reset or phase-align any detector integration, counter,
   or sample-ready state? If not, what establishes each network's initial
   integration phase after ROACH initialization?
8. Is a UDP payload built from one completed row and one coherent metadata
   record? Identify any asynchronous FIFO, double buffer, DMA, interrupt, or
   clock-domain crossing that can associate adjacent integration and metadata
   states.
9. Does one accepted UDP packet create exactly one NetCDF row? State the row
   indexing rule and behavior for loss, duplication, reordering, partial
   packets, process restart, counter wrap, and file boundaries.
10. At what event and on which clock is `Data.Toltec.RecvTime[n]` recorded?
    Is it captured before or after parsing, buffering, ordering, and NetCDF
    insertion, and is it guaranteed to refer to the same accepted packet as
    row `n`?
11. Did any relevant gateware, firmware/backend, packet schema, or NetCDF
    writer semantics differ among the three maps or among their networks?
    Identify each difference even if it is believed irrelevant.
12. Which answers are proved by the supplied source/specification, which are
    implementation intent only, and which remain unknown?

For reference, locally retained raw files exist only for map 148670. They show
`Header.Toltec.CompileTime=1731683195` on networks 0–5 and `1732640701` on
networks 7, 8, 9, 11, and 12. These integers are not accepted as source
revision mappings; please state what each identifies. No new raw detector data
is requested.

## Owner evidence-intake checklist

The owner should require the following before returning the response for
analysis.

### Identity and coverage

- [ ] Response explicitly lists all three observation numbers, map IDs, the
  frozen group ID, and all 11 networks.
- [ ] A 33-row map/network table names the deployed FPGA bitstream, firmware or
  backend, UDP schema, and host NetCDF-writer revision.
- [ ] Each revision is immutable: repository plus full commit hash or a
  versioned release/archive identifier with an archive SHA-256.
- [ ] `Header.Toltec.CompileTime` is mapped to the actual build/source identity
  or explicitly declared insufficient, including the two known 148670 values.
- [ ] Build date is recorded in ISO 8601 with timezone, and build host/toolchain
  versions are included where generated behavior can depend on them.

### Source-route integrity

- [ ] Every supplied file has its repository-relative path, byte size, and
  SHA-256 in a sorted manifest.
- [ ] The manifest also covers generated bitstream/firmware images and the UDP
  or register-schema artifact actually deployed, if retained.
- [ ] Build instructions or a build manifest bind source commit, generated
  outputs, tool versions, and compile-time identifier.
- [ ] Dirty-tree state at build time is stated. Any local patch is supplied as
  a file with its own SHA-256 and base commit.
- [ ] Source citations identify module/function/signal names for each answer,
  including clock domains and synchronizers where relevant.

### Specification-route authority

- [ ] Document title, stable version, author, technical approver, creation
  date, approval date, and scope are present.
- [ ] Approval date and all timestamps use ISO 8601 with timezone.
- [ ] The document file has byte size and SHA-256, and any revision history is
  included.
- [ ] Each answer cites a section, diagram, table, or requirement identifier.
- [ ] The approver confirms that the specification describes the deployed
  revisions for these exact maps, not only intended or current behavior.

### Semantic completeness

- [ ] All 12 questions are answered per distinct deployed revision.
- [ ] Integration bounds, intended event, fixed latency, counter semantics,
  PPS ordering, packet atomicity, NetCDF indexing, and `RecvTime` are each
  explicit.
- [ ] Proven behavior, design intent, recollection, inference, and unknowns are
  labeled separately.
- [ ] Known races, clock-domain crossings, buffering, dropped/reordered packet
  behavior, and revision differences are disclosed.
- [ ] The response does not convert the descriptive half-cadence labels into a
  physical-state identification or correction.

### Intake record

- [ ] The owner records responder name and role, delivery date/time, delivery
  mechanism, and the exact original filenames.
- [ ] The owner computes a SHA-256 manifest before any transformation and
  preserves the original response read-only.
- [ ] If confidentiality restricts source sharing, the authoritative
  specification route is used; screenshots or uncited prose do not replace a
  versioned source/specification artifact.
- [ ] Missing revision mapping or any unresolved event-order question is
  recorded as unavailable and returns SCI-ALIGN-001 to STOP rather than being
  filled by assumption.

## Local authority inventory used to prepare this request

- `validation/sci_align_001_acquisition_event_semantics_2026-08-08/REPORT.md`:
  accepted STOP result and surviving alternatives.
- `validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json`:
  owner-supplied producer clarification, including shared references, PPS ISR,
  UDP-carried counters, and unavailable FPGA source.
- `validation/sci_align_001_align_p0_d005_sky_domain_2026-08-02/timestamp_semantics_inventory.csv`:
  retained 148670 compile identifiers and absence of event-semantic metadata.
- `include/citlali/core/engine/detail/sci_align_netcdf_input_contract.h` and
  `include/citlali/core/engine/detail/todproc_alignment_impl.h`: delivered
  NetCDF schema and consumer row ingestion.
- `include/citlali/core/pipeline/sci_align_contract.h`: Citlali's legacy
  timestamp reconstruction.
- `include/citlali/core/pipeline/timestream_alignment_helpers.h`: downstream
  common-grid assignment behavior.

These are consumer-side or descriptive authorities. None supplies the missing
producer event contract, so this request stops for owner action after
preparation.
