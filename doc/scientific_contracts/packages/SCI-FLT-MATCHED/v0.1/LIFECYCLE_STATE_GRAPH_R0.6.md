# Corrected product lifecycle state graph — r0.6

Status: normative micro-repair draft; owner review pending

Successful requested route:

```text
requested
  -> effective
  -> learned_candidate       # when Learn applies
  -> resolved
  -> applied
  -> realized
  -> complete_publication_candidate
  -> publication_decided
  -> published | not_produced
```

`not_requested` is an alternative initial disposition. `disabled` and
`unavailable` are pre-application dispositions where applicable. `failed` may
branch from Learn, Resolve, Apply, numerical realization, product closure,
publication evaluation, or publication action.

`complete_publication_candidate` means complete and ready for policy
evaluation, not already publication-eligible. The SCI-VAL decision artifact is
realized separately from the FLT product.
