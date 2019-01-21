# SNOPT7.jl
Julia interface for SNOPT7

Still needs work, but currently works with:
- Julia 1.0
- JuMP 0.19-beta2
- MathOptInterface 0.8

- SNOPT 7.7 (including [trial libraries](ccom.ucsd.edu/~optimizers))

There are two examples, one using JuMP/MOI and the other using the "SNOPT"-like interface.

If you're using the trial libraries, set `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (linux) to the location of the libraries.
