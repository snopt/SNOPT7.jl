module SNOPT7

using SparseArrays

export initialize, snopt!, readOptions, setOption!
export snoptWorkspace

function __init__()
    # Set up library path
end

const libsnopt7 = "libsnopt7"

mutable struct snoptWorkspace
    status::Int

    # Workspace
    leniw::Int
    lenrw::Int
    iw::Vector{Int32}
    rw::Vector{Float64}

    leniu::Int
    lenru::Int
    iu::Vector{Int32}
    ru::Vector{Float64}

    x::Vector{Float64}
    lambda::Vector{Float64}
    obj_val::Float64

    num_inf::Int
    sum_inf::Float64

    iterations::Int
    major_itns::Int
    run_time::Float64

    function snoptWorkspace(leniw::Int,lenrw::Int)
        prob = new(0,leniw, lenrw,
                   zeros(Int,leniw), zeros(Float64,lenrw), 0, 0)
        finalizer(freeWorkspace!,prob)
        prob
    end
end

function freeWorkspace!(prob::snoptWorkspace)
    ccall((:f_snend, libsnopt7),
          Cvoid, (Ptr{Cint}, Cint, Ptr{Float64}, Cint),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
end

# Exit codes
SNOPT_status = Dict(
    1=>:Solve_Succeeded,
    2=>:Feasible_Point_Found,
    3=>:Solved_To_Acceptable_Level,
    4=>:Solved_To_Acceptable_Level,
    5=>:Solved_To_Acceptable_Level,
    6=>:Solved_To_Acceptable_Level,
    11=>:Infeasible_Problem_Detected,
    12=>:Infeasible_Problem_Detected,
    13=>:Infeasible_Problem_Detected,
    14=>:Infeasible_Problem_Detected,
    15=>:Infeasible_Problem_Detected,
    16=>:Infeasible_Problem_Detected,
    21=>:Unbounded_Problem_Detected,
    22=>:Unbounded_Problem_Detected,
    31=>:Maximum_Iterations_Exceeded,
    32=>:Maximum_Iterations_Exceeded,
    34=>:Maximum_CpuTime_Exceeded,
    41=>:Numerical_Difficulties,
    42=>:Numerical_Difficulties,
    43=>:Numerical_Difficulties,
    44=>:Numerical_Difficulties,
    45=>:Numerical_Difficulties,
    71=>:User_Requested_Stop,
    72=>:User_Requested_Stop,
    73=>:User_Requested_Stop,
    74=>:User_Requested_Stop,
    81=>:Insufficient_Memory,
    82=>:Insufficient_Memory,
    83=>:Insufficient_Memory,
    91=>:Invalid_Problem_Definition,
    92=>:Invalid_Problem_Definition,
    999=>:Internal_Error)

# Callbacks
function obj_wrapper!(mode_::Ptr{Cint}, nnobj::Cint, x_::Ptr{Float64},
                      f_::Ptr{Float64}, g_::Ptr{Float64},
                      status::Cint)
    x    = unsafe_wrap(Array, x_, Int(nnobj))
    mode = unsafe_load(mode_)

    if mode == 0 || mode == 2
        obj = convert(Float64, eval_f(x)) :: Float64
        unsafe_store!(f_,obj)
    end

    if mode == 1 || mode == 2
        g = unsafe_wrap(Array,g_,Int(nnobj))
        eval_grad_f(x,g)
    end
    return
end

function con_wrapper!(mode_::Ptr{Cint}, nncon::Cint, nnjac::Cint, negcon::Cint,
                      x_::Ptr{Float64}, c_::Ptr{Float64}, J_::Ptr{Float64},
                      status::Cint)

    x    = unsafe_wrap(Array, x_, Int(nnjac))
    mode = unsafe_load(mode_)

    if mode == 0 || mode == 2
        c = unsafe_wrap(Array, c_, Int(nncon))
        eval_g(x, c)
    end

    if mode == 1 || mode == 2
        J = unsafe_wrap(Array,J_,Int(negcon))
        eval_jac_g(x, J)
    end
    return
end


# SNOPT7 routines
function initialize(printfile::String, summfile::String)
    prob = snoptWorkspace(30500,3000)

    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{UInt8}, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printfile, length(printfile), summfile, length(summfile),
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end

function initialize(printfile::String, summfile::String,
                    leniw::Int, lenrw::Int)
    prob = snoptWorkspace(leniw, lenrw)

    ccall((:f_sninitx, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cint,
           Ptr{UInt8}, Cint, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          printfile, plen, prob.iprint, summfile, slen, prob.isumm,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return prob
end

function readOptions(prob::snoptWorkspace, specsfile::String)
    status = [0]
    ccall((:f_snspecf, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          specsfile, length(specsfile), status,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    prob.status = status[1]
    return Int(prob.status)
end

function snopt!(prob::snoptWorkspace, start::String, name::String,
                m::Int, n::Int, nnCon::Int, nnObj::Int, nnJac::Int,
                fObj::Float64, iObj::Int,
                confun::Function, objfun::Function,
                #eval_f::Function, eval_grad_f::Function,
                #eval_g::Function, eval_jac_g::Function,
                J::SparseMatrixCSC, bl::Vector{Float64}, bu::Vector{Float64},
                hs::Vector{Int}, x::Vector{Float64})

    @assert n+m == length(x) == length(bl) == length(bu)
    @assert n+m == length(hs)

    prob.iu = [0]
    prob.ru = [0.]

    prob.x      = copy(x)
    prob.lambda = zeros(Float64,n+m)
    pi          = zeros(Float64,m)

    obj_callback = @cfunction($objfun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}))
    con_callback = @cfunction($confun, Cvoid,
                             (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble},
                              Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}))

    valJ = J.nzval
    indJ = convert(Array{Cint}, J.rowval)
    locJ = convert(Array{Cint}, J.colptr)
    neJ  = length(valJ)

    status  = [0]
    nS      = [0]
    nInf    = [0]
    sInf    = [0.0]
    obj_val = [0.0]
    miniw   = [0]
    minrw   = [0]

    ccall((:f_snoptb, libsnopt7), Cvoid,
          (Ptr{UInt8}, Ptr{UInt8},
           Cint, Cint, Cint, Cint, Cint, Cint,
           Cint, Cdouble,
           Ptr{Cvoid}, Ptr{Cvoid},
           Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint},
           Ptr{Float64}, Ptr{Float64}, Ptr{Cint},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cint}, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint,
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          start, name, m, n, neJ, nnCon, nnObj, nnJac,
          iObj, fObj,
          con_callback, obj_callback,
          valJ, indJ, locJ,
          bl, bu, hs, prob.x, pi, prob.lambda,
          status, nS, nInf, sInf, obj_val,
          miniw, minrw,
          prob.iu, prob.leniu, prob.ru, prob.lenru,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)

    prob.status  = status[1]
    prob.obj_val = obj_val[1]

    prob.num_inf = nInf[1]
    prob.sum_inf = sInf[1]

    prob.iterations = prob.iw[421]
    prob.major_itns = prob.iw[422]

    prob.run_time   = prob.rw[462]

    return Int(prob.status)
end

function setOption!(prob::snoptWorkspace, optstring::String)
    # Set SNOPT7 option via string
    if !isascii(optstring)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snset, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

function setOption!(prob::snoptWorkspace, keyword::String, value::Int)
    # Set SNOPT7 integer option
    if !isascii(keyword)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snseti, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cint, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

function setOption!(prob::snoptWorkspace, keyword::String, value::Float64)
    # Set SNOPT7 real option
    if !isascii(keyword)
        error("SNOPT7: Non-ASCII parameters not supported")
    end

    errors = [0]
    ccall((:f_snseti, libsnopt7), Cvoid,
          (Ptr{UInt8}, Cint, Cdouble, Ptr{Cint},
           Ptr{Cint}, Cint, Ptr{Cdouble}, Cint),
          optstring, length(optstring), value, errors,
          prob.iw, prob.leniw, prob.rw, prob.lenrw)
    return errors[1]
end

include("MOIWrapper.jl")

end # module
