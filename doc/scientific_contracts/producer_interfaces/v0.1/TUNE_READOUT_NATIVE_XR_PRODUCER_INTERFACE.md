# Tune/Readout Native Paired-`x/r` Producer Interface

Interface identity: `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1`

Status: owner-decision-complete candidate; exact artifact approval pending

Prepared: `2026-08-23`

Scientific owner: Grant Wilson

Interface owner: Tune/readout scientific producer

Consumers: SCI-ALIGN directly; SCI-RTC through the exact SCI-ALIGN relation

## 1. Purpose And Ownership

This package-neutral record defines the producer interface by which native
measured readout occurrences become exact paired detector-coordinate
occurrences before SCI-ALIGN. It binds the external producer authority already
required by the frozen SCI-ALIGN and SCI-RTC contracts; it does not introduce
a new transformation, coordinate convention, CAL acquisition boundary, or
runtime provenance payload. A consumer references the authoritative
Tune/readout record rather than copying its contents.

The ordinary order is

\[
(I,Q)^{\rm acq}
\longrightarrow \text{Tune/readout}
\longrightarrow (x,r)^{\rm acq}
\longrightarrow \text{SCI-ALIGN}
\longrightarrow (x,r)^A
\longrightarrow \text{SCI-RTC}.
\]

Tune/readout owns the native \(I,Q\rightarrow x,r\) transformation and its
coordinate meaning. SCI-ALIGN owns the native-to-aligned occurrence relation.
SCI-RTC begins with the admitted aligned pair and owns only its later
conditioning. SCI-CAL neither consumes this interface directly nor
reinterprets the native readout; the interface identity may remain available
to CAL only through transitive provenance.

## 2. Composed Frozen Authority

This interface composes, without superseding or reopening:

- SCI-ALIGN v0.1/r0.3 definitions lines 6--20 and 34--41,
  assumption `SCI-ALIGN-ASM-005`, and requirements
  `SCI-ALIGN-REQ-001`, `003--007`, and `050`;
- SCI-RTC v0.1/r0.12 definitions `SCI-RTC-DEF-003` and
  `SCI-RTC-DEF-031`, and requirements `SCI-RTC-REQ-001--003`,
  `006--008`, `083--085`, `103`, `134`, `137`, and `139`; and
- the owner-approved clarification `WP2-FOLLOWUP-D011`.

If a successor changes any cited meaning, it shall provide an explicit
semantic mapping and compatibility disposition rather than silently editing
this interface.

## 3. Static Producer-Interface Authority

The versioned Tune/readout interface specification shall define or
authoritatively reference:

1. producer identity, interface identity and revision, transform identity,
   and the schema or record family by which an instance is resolved;
2. the native measured readout coordinates admitted as the transform input;
3. the output \(x\) and \(r\) coordinate meanings, units or scales, sign,
   reference point or baseline, normalization, and metric where applicable;
4. the exact transform representation, including its local Jacobian or
   nonlinear equivalent where required to interpret the admitted output;
5. applicability and validity or linearity domains, Tune/mapping-revision
   boundaries, epoch meaning, and uncertainty or typed-unavailability rules;
6. the stable identity relation among observation, Tune, network/interface,
   tone or channel, detector occurrence, native readout occurrence, and the
   resulting paired \(x/r\) occurrence; and
7. the required consumer guarantee, compatibility test, runtime binding rule,
   failure semantics, and provenance fields.

This record does not standardize a new sign, reference, or normalization.
Each realized instance carries the values owned by its exact upstream
producer authority.

## 4. Observation-Instance Realization

For each admitted observation and native occurrence, the runtime binding shall
resolve:

- the exact observation, Tune, producer-interface, and mapping-record
  reference;
- the network/interface and stable tone, channel, or admitted row identity;
- the exact detector occurrence and its owning detector-association reference;
  any selected APT binding remains separately owned;
- the parent measured-readout occurrence or exact parent-record relation;
- one exact paired native \((x,r)^{\rm acq}\) occurrence identity;
- the native event-time and acquisition/integration-support relation required
  by SCI-ALIGN; and
- coordinate-specific numerical availability and validity for \(x\) and
  \(r\), independent of their common pair identity.

The exact producer/mapping reference resolves the transform, units or scales,
sign, reference, normalization, epoch, applicability domain, and available
uncertainty state. Those facts may reside in the referenced Tune/readout
record; this interface does not require them to be copied into every
occurrence or into another sidecar.

The observation's numerical payload and selected Tune record are execution
instances. They need not be embedded in the frozen scientific-contract
repository.

## 5. Pair And Mapping Invariants

1. The native \(x\) and \(r\) outputs form one ordered paired occurrence from
   the same admitted native readout occurrence under one exact mapping
   revision.
2. Pair identity, occurrence identity, and mapping identity exist
   independently of coordinate-specific numerical validity.
3. A transient storage column or row position is not cross-artifact detector
   identity.
4. No consumer may synthesize a missing partner, copy one coordinate into the
   other, zero-fill it, infer the transform from shape or numerical range, or
   cross a Tune/mapping-revision boundary without separately named authority.
5. SCI-ALIGN may map only the admitted native pair into its exact aligned pair
   relation. SCI-RTC consumes that aligned pair and shall not reapply either
   Tune/readout or ALIGN.
6. Neither native nor aligned \(x/r\) has calibrated, Stokes, or other science
   quantity meaning merely by passing this interface.

## 6. Failure And Availability

Missing or ambiguous required producer identity, mapping revision, pair
identity, detector/tone association, transform convention, sign, reference,
normalization, applicability, validity domain, or runtime association makes
the affected native pair unavailable for required SCI-ALIGN admission. A
malformed or incomplete aligned pair then fails SCI-RTC required-input
admission at the exact affected scope.

No downstream package may repair the failure by consulting implementation
defaults, matching array shape, comparing finite values, or reinterpreting the
native readout. Optional uncertainty components may be typed unavailable only
where the frozen consumer contract permits that state; required identity and
mapping facts may not be replaced by typed numerical availability.

## 7. Claim Boundary

This interface establishes candidate source-closure semantics only. It does
not establish that a producer supplies a conforming record, that any
observation instance is valid, that SCI-ALIGN or SCI-RTC implements the
interface, or that any numerical, observational, performance, production, or
MAP claim has been demonstrated.
