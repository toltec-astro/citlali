# Codex Notes

## Build/Test Environment

The required local toolchain for compiling `citlali` has not been installed on this machine.

For Codex work in this repository:
- Do **not** attempt local configure/build commands.
- Do **not** run local compile/test commands as part of normal verification.
- Perform authoritative build and runtime validation on **Unity** only.
- If a build/test check is needed, request or use Unity.

## Practical Guidance

- Make code changes locally.
- Validate syntax/style with lightweight checks when possible.
- Run full configure/build/reduction tests on Unity before concluding correctness.
- Whenever editing Citlali, add or append to the current day's handoff note in `handoff/HANDOFF_YYYY-MM-DD.md` with the local timestamp before ending the session.

## Memo Formatting

- For TolTEC/Citlali memo-style LaTeX documents, use the technical memo template from `/Users/gwilson/GitHub/toltec-memoranda/templates/technical_memo_template.tex`.
- Use `G. Wilson \& Codex` as the author when Codex materially drafts or edits the memo.
