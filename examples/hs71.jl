# HS71
# Polynomial objective and constraints
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
#
# Initial x = (1,5,5,1)
#  (1.000..., 4.743..., 3.821..., 1.379...)

using JuMP, SNOPT7

m = Model(with_optimizer(SNOPT7.Optimizer, print_level=0,system_information="yes"))

x0 = [1,5,5,1]

@variable(m, 1 <= x[i=1:4] <= 5)
@NLobjective(m, Min, x[1]*x[4]*(x[1]+x[2]+x[3]) + x[3])
@NLconstraint(m, x[1]*x[2]*x[3]*x[4] >= 25)
@NLconstraint(m, sum(x[i]^2 for i=1:4) == 40)

JuMP.optimize!(m)

println(JuMP.value.(x))

objval = JuMP.objective_value(m)
println("Final objective: $objval")

println(JuMP.termination_status(m))
