# Scientific Method Notes

This directory is a small, durable library of mathematical and statistical
methods used by Citlali. It is not a collection of audit narratives and it is
not intended to document every calculation in the pipeline.

## When A Note Is Warranted

Create a note when a method is non-obvious or non-standard, materially affects
scientific interpretation, or needs assumptions and estimator properties that
do not belong in a usage guide. A familiar identity with an adequate canonical
reference usually needs only a citation. An implementation detail without a
scientific consequence belongs in engineering documentation or code.

Most notes should fit in one or two pages. Longer treatment is appropriate
only when the scientific argument genuinely requires it.

These notes are rendered as GitHub-flavored Markdown. Use `$...$` for inline
mathematics and `$$...$$` for display mathematics; do not use LaTeX document
delimiters such as `\(...\)` or `\[...\]`, which GitHub does not render.

## One Technique, One Explanation

Each method receives a stable ID of the form
`SCI-METHOD-<SHORT-NAME>-NNN`. A note owns the method's equations, assumptions,
properties, limitations, and validation links. Product guides, configuration
guides, audit reports, and pipeline-stage documentation refer to that ID and
link to the note instead of restating the derivation.

When another stage adopts an unchanged method, add that consumer to the
registry and, if useful, to the note's consumer list. Do not create a second
note. A materially incompatible definition receives a new method ID and an
explicit supersession relationship; accepted historical meaning is never
silently rewritten.

## Method Registry

Register a method only alongside a stable implementation or accepted contract;
do not reserve IDs for speculative work.

| Method ID | Note | Scientific purpose | Consumers | Status |
| --- | --- | --- | --- | --- |
| `SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001` | [Ordinary positive-coefficient map normalization](SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001.md) | Normalize ordinary naive signal and matched companions while preserving nonprecision semantics | Naive observation maps; admitted observation coadds; matched kernel and declared linear companions | `validated_bounded` |

## Required Reference Form

At first use in a consuming document, cite both the stable ID and link, for
example:

```markdown
The diagnostic uses
[SCI-METHOD-EXAMPLE-001](../science/SCI-METHOD-EXAMPLE-001.md).
```

Subsequent references in the same document may use the stable ID alone. Source
metadata may record the ID without a path so that document reorganization does
not change the scientific identity.

Use [`METHOD_NOTE_TEMPLATE.md`](METHOD_NOTE_TEMPLATE.md) as a starting point
and delete sections that add no value.
