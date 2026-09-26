# Local-data RTC/PTC purge handoff — 2026-09-26

The owner's later September 26 directive supersedes **all earlier saved RTC/PTC
retention and restoration claims below**, including unique/shared parents and
producer-specific restart exceptions. Across the whole local_data tree, 221,249
identified inactive outputs were removed: 356,819,904,701 logical bytes. Mixed
NetCDF non-target contents were first separated and verified (1,311 companions,
5.194 GB). Actual deletion-interval free-space gain was 353,300,783,104 bytes;
logical tree size changed from 970.754 to 619.129 GB. No failures or live exceptions.
All 329,137 original non-target file identities and all 20 symlinks are unchanged.

Maps, scientific conclusions, original manifests, raw/tune/calibration inputs and
independent matrices remain. Durable compact recovery records were committed before
removal at 6a0b75236c8880e0ae588f9b21c8766ebaa4eb9c in the permanent Git repository.
Deleted routes now require explicit upstream reprocessing; old full-payload seals
and canonical-copy availability are historical, not current. Current purge and
archive-reader notices make this absence explicit without automatic regeneration.
Six purge tests and eight archive-reader tests pass; independent safety and inventory
reviews bind the operation. No scientific algorithms or defaults changed.

The inventory identifies 21 sky packages complete against saved input inventories,
five with expected-inventory/calibration readiness unverified, and partial 123424.
Proposed citlali-work root, 200 GB working budget and 1.830/9.410/26.963 GB input
corpora await owner selection; they are not a quota or a limit on retained historical
data. CUT3 and the next observation remain parked. No Unity access, merge or push.
See validation/local_data_rtc_ptc_purge_2026-09-26/REPORT.md and COMPLETION.json,
and handoff/LOCAL_DATA_RTC_PTC_PURGE_2026-09-26.md. Stop after this cleanup/report.


Exact execution source: `6a0b75236c8880e0ae588f9b21c8766ebaa4eb9c`.
Permanent Git common directory: `/Users/gwilson/GitHub/citlali/.git`.
Owned branch: `codex/map-successor-first-forward-001`.
Do not rerun the deletion command. Completion/receipt and current root index
identify intentionally absent files; reading reports must not rebuild them.
