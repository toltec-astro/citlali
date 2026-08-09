# Handoff to SCI-TEL-INPUT-001

Date: 2026-08-08

This handoff does not launch or broaden the 20-ms TolTECA/telescope-file
ingress audit.

- Detector `Data.Toltec.Ts[:,4]` is the TolTEC raw `PpsTime` clock-tick field.
  It must not be conflated with telescope
  `Data.TelescopeBackend.PpsTime`.
- The detector acquisition audit establishes delivered `D[n]/Ts[n]` row
  lineage but not the physical integration event represented by that row.
  Detector time therefore cannot serve as an absolute physical oracle for the
  telescope-ingress audit without later producer authority.
- No telescope row, timestamp, interpolation, 20-ms association, or
  recomputation was inspected in this audit.
- The three descriptive same-T0 labels remain non-physical and must not be
  imported into SCI-TEL-INPUT-001 as a correction or prior.
