### Cahn-Hillard example with constant concentration

using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using StaticArrays

import Pkg as JLPkg
JLPkg.develop(path = joinpath(JLPkg.devdir(), "LiquidCrystals"))
using LiquidCrystals

# Defining the constants and parameters in our binary mixture
const R = 8.31446261815324 # Gas constant in J mol^-1 K^-1
const T = 673 # Temperature in K

# Flory-Huggins interaction parameters
const chi = 20.71 * T
const K = 3e-14
const Da = 1e-4 * exp(-3e5 / (R * T))
const Db = 2e-5 * exp(- 3e5/ (R * T))

# Mean value of the concentration of one species
const c_0 = 0.5
# Size fluctuations in concentration in the initial state
const del_c = 0.25
const Mc = 1e-23 #(Da / (R * T)) * (c_0 + (Db/Da) * (1.0 - c_0)) * c_0 * (1 - c_0)
print("Mc", Mc)

# Dimensions of the space discretization
const Nx = Ny = 50

# We now randomly initialize the concentration field 

phi_0 = c_0 .+ 2*del_c .* rand(Nx, Ny) .- del_c

#= Numerical solution
    Set the spatial discretization dimensions along each axis and the corresponding finite-differences laplacian.
=#
const LC = LiquidCrystals
const dx = dy = 2e-9
del_r = (dx, dy)
Del = LC.CenteredDifference{LC.Laplacian}(del_r);

#=
    Select Boundary Conditions
=#

#=
    Build memory caches for Numerical integration
=#
bc = LC.BoxBC(
    LC.PeriodicAxisBC{LC.XAxis}(0.0),
    LC.PeriodicAxisBC{LC.YAxis}(0.0)
)
#=
bc = LC.BoxBC(
    LC.NeumannAxisBC{LC.XAxis}(0.0),
    LC.NeumannAxisBC{LC.YAxis}(0.0)
)
=#
E_phi, P_phi = LC.build_caches(Del, bc, phi_0)
E_mu, P_mu = LC.build_caches(Del, bc, phi_0)

#=
        Define discretized ODE as a PDE function
=#
function cahn_hillard(du, u, p, t)
    Del, bc, E_phi, P_phi, E_mu, P_mu = p
    RT = R * T

    phi = @view u[:, :]
    dphi = @view du[:, :]
    # Extended matrix with ghosts at the edges
    E_phi = LC.apply_BCs(Del, phi, bc, E_phi)
    # Apply Laplacian Operator on phi
    mul!(P_phi, Del, E_phi)

    # Compute chemical potential
    Phi = @view(P_phi[2:end-1, 2:end-1])
    Mu = @. -K * Phi + chi * (1 - 2 * phi) + RT * log( phi / (1 - phi))

    E_mu = LC.apply_BCs(Del, Mu, bc, E_mu)

    # Apply Laplacian Operator on mu
    mul!(P_mu, Del, E_mu)

    @. dphi = Mc * @view(P_mu[2:end-1, 2:end-1])

    return du
end

#=
    Set the time upper bound for evolving the dynamics and the number of snapshots we wish to save
=#

t_f = 55000
Nt = 100

#=
    Set the parameters and bundle them together with the ODE function, initial condition, time span in a 'ODEProblem'.
=#

params = (Del, bc, E_phi, P_phi, E_mu, P_mu);
prob = ODEProblem(cahn_hillard, phi_0, (0.0, t_f), params);

println("Parameter Initialization Done")

#=
    Solve the ODE.
=#

sol = solve(prob, Tsit5(), progress = true, saveat = t_f/Nt, save_start = true)

phi_initial = sol.u[1]
phi_final   = sol.u[end]

fig = Figure(resolution = (900, 400))

ax1 = Axis(fig[1, 1], title = "ϕ at t = 0")
hm1 = heatmap!(ax1, phi_initial, colormap = :viridis)
Colorbar(fig[1, 1, Right()], hm1)

ax2 = Axis(fig[1, 2], title = "ϕ at t = t_f")
hm2 = heatmap!(ax2, phi_final, colormap = :viridis)
Colorbar(fig[1, 2, Right()], hm2)

fig