# Pre-science implementation/fixture repairs

No trajectories or RTC diagnostic executions have started. Initial native build
caught a YAML iterator-value assignment; changed it to indexed Node binding.
Initial Python fixtures expected RuntimeError although the established require
API raises ValueError; the implementation correctly rejected both invalid cases.
The new unchanged-detector fixture used numerical equality across unavailable
NaNs; corrected the assertion to exact payload-byte equality, retaining strict
identity including signed zero/NaN bits. Initial fixture failures are preserved.
No scientific tolerance, filter, membership or support choice was changed.
