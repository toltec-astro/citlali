# Lifecycle amendment — r0.5

Status: normative closure amendment

The complete vocabulary is:

'not_requested', 'requested', 'effective', 'disabled', 'unavailable',
'learned_candidate', 'resolved', 'applied',
'complete_publication_candidate', 'publication_decided', 'realized',
'published', 'failed', 'not_produced', and 'superseded'.

'learned_candidate' is candidate state not yet authorized or frozen.
'resolved' is the immutable method/state generation before Apply.
'realized' records immutable values and outcome provenance, not publication.
'publication_decided' is the authorized accept/reject decision on an immutable
candidate. 'published' is the authorized exposure of that immutable product.
These states never collapse merely because their values or timestamps agree.
