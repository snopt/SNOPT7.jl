# SNOPT7.jl
Julia interface for SNOPT7

Still needs work, but currently works with:
- Julia 1.4.2
- JuMP 0.21.2
- MathOptInterface 0.9.14

- SNOPT 7.7 (including [trial libraries](http://ccom.ucsd.edu/~optimizers))

There are two examples, one using JuMP/MOI and the other using the "SNOPT"-like interface.

If you're using the trial libraries, set `DL_LOAD_PATH` (macOS) or `LD_LIBRARY_PATH` (linux) to the location of the libraries.
