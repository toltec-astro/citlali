# Phase A independent review and repair

Initial review candidate 637e2adce4ca65e01510106273aa8e80c96f3433,
tree 481f216d230db82a7c0aad6847cbdedaba95ed5d: implementation ownership and
bounded repository scope pass; scientific/failure publication required repair.
Reviewer reproduced a T1 processing-guard failure that advertised the prior
successful state. No real science was run.

Repair publishes the current failed state, atomically exposes READY files, and
separates initial source rejection from processing failure with diagnostic M0
availability. Process startup records now count instrumented actual attempts,
not the planned N. Tests cover failure after a current measurement and N=0/1/3/5/10.
Exact repaired-SHA recheck is required before launch. No numerical change.

Recheck of 516a283bc76bbfe1e259d92357bd191819b2782c (tree
5b465b63371287222119ef4ef50b56d947af6ac7) found a later cleanup/publication
exception could retain ready=true. All caught cycle exceptions now force
ready=false and overwrite the current failed state before READY publication.
Added mocked cleanup failures at every positive N. All57 focused tests pass;
no real-data attempt consumed. Ownership and bounded scope passed both reviews.
