using SNOPT7
using SparseArrays

function objfun(mode_::Ptr{Cint}, n_::Ptr{Cint}, x_::Ptr{Float64},
                f_::Ptr{Float64}, g_::Ptr{Float64},
                status::Ptr{Cint})
    mode = unsafe_load(mode_)
    n    = unsafe_load(n_)

    x    = unsafe_wrap(Array, x_, Int(n))

    if mode == 0 || mode == 2
        obj = x[1] + x[2]
        unsafe_store!(f_,obj)
    end

    if mode == 1 || mode == 2
        g = unsafe_wrap(Array,g_, Int(n))
        g[1] = 1.0
        g[2] = 1.0
    end
    return
end

function confun(mode_::Ptr{Cint},
                nncon_::Ptr{Cint}, nnjac_::Ptr{Cint}, negcon_::Ptr{Cint},
                x_::Ptr{Float64}, c_::Ptr{Float64}, J_::Ptr{Float64},
                status::Ptr{Cint})
    mode   = unsafe_load(mode_)
    nncon  = unsafe_load(nncon_)
    nnjac  = unsafe_load(nnjac_)
    negcon = unsafe_load(negcon_)

    x    = unsafe_wrap(Array, x_, Int(nnjac))
    if mode == 0 || mode == 2
        c = unsafe_wrap(Array, c_, Int(nncon))
        c[1] = x[1]^3 - x[2]

    end

    if mode == 1 || mode == 2
        J = unsafe_wrap(Array,J_, Int(negcon))
        J[1] = 3.0*x[1]^2
        J[2] = -1.0

    end
    return

end

function eval_f(x)
    return (x[1] - 2.0)^2 + x[2]^2
end

function eval_grad_f(x, grad)
    grad[1] = 2.0*(x[1] - 2.0)
    grad[2] = 2.0*x[2]
end

function eval_g(x,c)
    c[1] = (1.0 - x[1])^3  - x[2]
end

function eval_jac_g(x,J)
    J[1] = -3.0*(1-x[1])^2
    J[2] = -1.0
end


inf  = 1.0e20
n    = 2
m    = 1
iObj = 0

hs = zeros(Int,n+m)
bl = [ -inf, 1., 0.0 ]
bu = [  inf, inf, inf ]
x  = [ 0. , 3., 1. ]

nnCon = 1
nnJac = 2
nnObj = 2

J = [ -100 -1.0 ]
J = sparse(J)

prob = initialize("hs13.out", "screen")
info = readOptions(prob, "hs13.spc")
info = snopt!(prob, "cold", "hs13",
              m, n, nnCon, nnObj, nnJac, 0., iObj,
              confun, objfun,
              J, bl, bu, hs, x)

println("initial x = ", x)
println("final x = ", prob.x)
println("final obj = ", prob.obj_val)
println("SNOPT status = ", prob.status)

println("sum of inf: = ", prob.sum_inf)

println("num iterations = ", prob.iterations)
println("num major iterations = ", prob.major_itns)
println("solve time = ", prob.run_time)
