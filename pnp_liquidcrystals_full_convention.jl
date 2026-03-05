""" 
Steady Poisson–Nernst–Planck (PNP) in a 2D pore using LiquidCrystals.jl operators/BC conventions.

This script is aligned with the finite-difference operator and boundary-condition
framework in LiquidCrystals.jl (as in `operators.jl`):

  - Operators: `CenteredDifference{Laplacian}`, `{Gradient}`, `{Divergence}`
    acting on *extended* arrays (with 1 ghost layer on each side).
  - BCs: `BoxBC((AxisBC_x, AxisBC_y))` and `apply_BCs(op, u, box, extended)`
    to fill ghost cells consistently.

We implement the nonlinear, coupled Nernst–Planck no-wall-flux condition
  n·(∇c_± ± Ṽ c_± ∇φ̃) = 0
on y-walls by defining a custom `AxisBC` that (i) has access to the already
BC-filled extended potential `φ̃` and (ii) computes the outward normal derivative
of φ̃ using the same centered (ghost/interior) stencil used by the package's
NeumannAxisBC.

Scaling matches your notes:
  x̃ = x/a, ỹ = y/a,  c̃ = c/c0,  φ̃ = φ/ΔV.
  Ṽ = eΔV/(kBT), and Poisson: ∇̃² φ̃ + (κa)^2/(2 Ṽ) (c̃_+ - c̃_-) = 0.

BCs:
  x̃=0:   c̃_±=1,  φ̃=0
  x̃=L̃:  c̃_±=1,  φ̃=1
  ỹ=0,1: ∂_n φ̃ = 0  (set σ̃=0 here; easy to extend)
          n·(∇c̃_± ± Ṽ c̃_± ∇φ̃)=0

Notes:
  * Newton on the fully coupled system can be stiff for large κa or Ṽ.
    Continuation in Ṽ (or κa) is recommended if convergence is problematic.
"""

using LinearAlgebra
using StaticArrays
using NonlinearSolve

import LiquidCrystals
import LiquidCrystals: CenteredDifference, Laplacian, Gradient, Divergence
import LiquidCrystals: BoxBC, DirichletAxisBC, NeumannAxisBC, AxisBC, XAxis, YAxis
import LiquidCrystals: build_caches, apply_BCs, ghost_ranges, boundary_ranges, get_dr


# -------------------------
# Parameters / discretization
# -------------------------

const Ltilde  = 5.0
const Vtilde  = 5.0
const kappa_a = 5.0
const sigma_tilde = 0.0   # for ∂_n φ̃ = -σ̃ at y=0 and ∂_n φ̃ = +σ̃ at y=1 (outward normals)

const Nx = 160
const Ny = 40

const dx = Ltilde/(Nx-1)
const dy = 1.0/(Ny-1)

# Unknowns on the *interior* (no ghosts)
cplus  = ones(Float64, Nx, Ny)
cminus = ones(Float64, Nx, Ny)
phi    = [ (i-1)*dx/Ltilde for i in 1:Nx, j in 1:Ny ]   # linear in x


# -------------------------
# Operators
# -------------------------
lap = CenteredDifference{Laplacian}((dx, dy))
grad = CenteredDifference{Gradient}((dx, dy))
div  = CenteredDifference{Divergence}((dx, dy))


# -------------------------
# Custom coupled BC: no Nernst–Planck normal flux on Y walls
# -------------------------

"""No-flux Nernst–Planck boundary condition for a scalar concentration on an Axis.

Imposes, on the wall with outward unit normal n:

    ∂_n c + zV * c * ∂_n φ = 0,

where zV is (+Ṽ) for cations and (−Ṽ) for anions, assuming φ is the dimensionless
potential φ̃ (scaled by ΔV).

This BC is nonlinear (depends on c) and coupled (depends on φ), so we implement it
by filling the ghost layer of c using the same convention as NeumannAxisBC:

  low wall (outward normal −ŷ):  ∂_n c ≈ (c_g - c_b)/(2dy)
  high wall (outward normal +ŷ): ∂_n c ≈ (c_b - c_g)/(2dy)

We compute ∂_n φ consistently from the already-filled extended potential φ_ext.
"""
struct NoFluxNernstPlanckAxisBC{A<:LiquidCrystals.Axis, T<:Real} <: AxisBC
    zV::T
    phi_ext_ref::Base.RefValue{AbstractMatrix{T}}
end

function NoFluxNernstPlanckAxisBC{A}(zV::T, phi_ext_ref::Base.RefValue{AbstractMatrix{T}}) where {A<:LiquidCrystals.Axis, T<:Real}
    return NoFluxNernstPlanckAxisBC{A,T}(zV, phi_ext_ref)
end

"""Apply the coupled no-flux BC by writing ghost values into `extended`.

Signature matches LiquidCrystals.jl dispatch: apply_BCs(op, u, bc, extended).
We read φ from `bc.phi_ext_ref[]`.
"""
function LiquidCrystals.apply_BCs(op::CenteredDifference{N,T,Tag}, u::AbstractMatrix{T}, bc::NoFluxNernstPlanckAxisBC{A,T}, extended::AbstractMatrix{T}) where {N,T,Tag,A<:LiquidCrystals.Axis}
    # This BC is intended for Y walls only.
    axis = get_dr(A)
    @assert axis == 2 "NoFluxNernstPlanckAxisBC currently implemented for YAxis only"

    phi_ext = bc.phi_ext_ref[]
    dy_loc = op.dr[axis]

    sz = size(extended)
    # Ghost index ranges in extended coordinates
    lo_g, hi_g = ghost_ranges(sz, A)
    CI_lg = CartesianIndices(lo_g)
    CI_hg = CartesianIndices(hi_g)

    # Boundary-adjacent interior indices in u coordinates
    CI_lb_u, CI_hb_u = boundary_ranges(sz, NeumannAxisBC{A}(0.0))

    # Map u indices -> extended indices by + (1,1)
    shift = CartesianIndex(ntuple(_->1, Val(N))...)
    CI_lb_e = CI_lb_u .+ shift
    CI_hb_e = CI_hb_u .+ shift

    # Outward normal derivatives of phi (∂_n φ) computed using ghost/interior stencil
    # low (outward −ŷ):  ∂_n φ = (φ_g - φ_b)/(2dy)
    dphi_n_low  = (phi_ext[CI_lg] .- phi_ext[CI_lb_e]) ./ (2*dy_loc)
    # high (outward +ŷ): ∂_n φ = (φ_b - φ_g)/(2dy)
    dphi_n_high = (phi_ext[CI_hb_e] .- phi_ext[CI_hg]) ./ (2*dy_loc)

    # ∂_n c = - zV * c * ∂_n φ
    dc_n_low  = .-(bc.zV) .* u[CI_lb_u] .* dphi_n_low
    dc_n_high = .-(bc.zV) .* u[CI_hb_u] .* dphi_n_high

    # Fill ghosts using the same sign convention as NeumannAxisBC:
    # low:  c_g = c_b + 2 dy * ∂_n c
    # high: c_g = c_b - 2 dy * ∂_n c
    extended[CI_lg] .= u[CI_lb_u] .+ 2*dy_loc .* dc_n_low
    extended[CI_hg] .= u[CI_hb_u] .- 2*dy_loc .* dc_n_high

    return extended
end


# -------------------------
# BCs for φ, c+, c-
# -------------------------

bc_phi = BoxBC((
    DirichletAxisBC{XAxis}(0.0, 1.0),
    NeumannAxisBC{YAxis}(-sigma_tilde, +sigma_tilde),
))

# Build caches for φ first (we will share φ_ext with the NP BCs)
phi_ext, lap_phi_ext = build_caches(lap, bc_phi, phi)
phi_ext_ref = Ref{AbstractMatrix{Float64}}(phi_ext)

bc_cplus = BoxBC((
    DirichletAxisBC{XAxis}(1.0, 1.0),
    NoFluxNernstPlanckAxisBC{YAxis}(+Vtilde, phi_ext_ref),
))

bc_cminus = BoxBC((
    DirichletAxisBC{XAxis}(1.0, 1.0),
    NoFluxNernstPlanckAxisBC{YAxis}(-Vtilde, phi_ext_ref),
))


# Caches for concentrations and gradients/divergence
cplus_ext,  grad_cplus_ext  = build_caches(grad, bc_cplus,  cplus)
cminus_ext, grad_cminus_ext = build_caches(grad, bc_cminus, cminus)

# Cache for grad(phi)
_, grad_phi_ext = build_caches(grad, bc_phi, phi)

# Flux arrays (extended) and divergence buffers
Jp_ext = similar(grad_phi_ext)  # SVector-valued array
Jm_ext = similar(grad_phi_ext)
div_Jp_ext = zeros(Float64, size(phi_ext)...) 
div_Jm_ext = zeros(Float64, size(phi_ext)...)


# -------------------------
# Pack / unpack utilities
# -------------------------

const npts = Nx*Ny
@inline function pack_state(cplus, cminus, phi)
    U = Vector{Float64}(undef, 3npts)
    U[1:npts] .= vec(cplus)
    U[npts+1:2npts] .= vec(cminus)
    U[2npts+1:3npts] .= vec(phi)
    return U
end

@inline function unpack_state!(cplus, cminus, phi, U)
    cplus .= reshape(@view(U[1:npts]), Nx, Ny)
    cminus .= reshape(@view(U[npts+1:2npts]), Nx, Ny)
    phi   .= reshape(@view(U[2npts+1:3npts]), Nx, Ny)
    return nothing
end


# -------------------------
# Residual
# -------------------------

function residual!(F, U, p)
    unpack_state!(cplus, cminus, phi, U)

    # --- Potential: fill ghosts and compute Laplacian ---
    apply_BCs(lap, phi, bc_phi, phi_ext)
    mul!(lap_phi_ext, lap, phi_ext)

    # Also compute grad(phi) on the same extended potential
    mul!(grad_phi_ext, grad, phi_ext)
    # Update the reference used by the NP BCs (points to the same array, but safe)
    phi_ext_ref[] = phi_ext

    # --- Concentrations: fill ghosts using coupled BCs and compute gradients ---
    apply_BCs(grad, cplus,  bc_cplus,  cplus_ext)
    apply_BCs(grad, cminus, bc_cminus, cminus_ext)

    mul!(grad_cplus_ext,  grad, cplus_ext)
    mul!(grad_cminus_ext, grad, cminus_ext)

    # --- Build fluxes in extended domain at interior nodes ---
    # J± = -(∇c± ± Ṽ c± ∇φ)
    sz = size(phi_ext) .- 1
    CI = CartesianIndices(UnitRange.(2, sz))
    @inbounds for I in CI
        Jp_ext[I] = -(grad_cplus_ext[I]  + Vtilde * cplus_ext[I]  * grad_phi_ext[I])
        Jm_ext[I] = -(grad_cminus_ext[I] - Vtilde * cminus_ext[I] * grad_phi_ext[I])
    end

    # Divergence of flux
    fill!(div_Jp_ext, 0.0); fill!(div_Jm_ext, 0.0)
    mul!(div_Jp_ext, div, Jp_ext)
    mul!(div_Jm_ext, div, Jm_ext)

    # --- Assemble residuals on interior (non-ghost) grid ---
    # Map interior u indices (1:Nx,1:Ny) to extended indices (+1,+1)
    Rp = @view div_Jp_ext[2:end-1, 2:end-1]
    Rm = @view div_Jm_ext[2:end-1, 2:end-1]
    lap_phi_int = @view lap_phi_ext[2:end-1, 2:end-1]
    Rφ = similar(phi)
    Rφ .= lap_phi_int
    @. Rφ += 0.5*(kappa_a^2/Vtilde) * (cplus - cminus)

    F[1:npts] .= vec(Rp)
    F[npts+1:2npts] .= vec(Rm)
    F[2npts+1:3npts] .= vec(Rφ)
    return nothing
end


# -------------------------
# Solve
# -------------------------

U0 = pack_state(cplus, cminus, phi)
prob = NonlinearProblem(residual!, U0)

using FiniteDiff

prob = NonlinearProblem(residual!, U0)

sol = solve(prob,
            NewtonRaphson(; autodiff = false),  # IMPORTANT: Bool in your version
            jacobian = FiniteDiff.JacobianCache(U0),
            abstol=1e-8, reltol=1e-8, maxiters=40)

unpack_state!(cplus, cminus, phi, sol.u)

println("retcode = ", sol.retcode)
println("phi range: ", (minimum(phi), maximum(phi)))
println("cplus range: ", (minimum(cplus), maximum(cplus)))
println("cminus range: ", (minimum(cminus), maximum(cminus)))
