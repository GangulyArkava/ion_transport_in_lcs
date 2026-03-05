using OrdinaryDiffEq
using LinearAlgebra
using StaticArrays
import LiquidCrystals as LC

# ============================================================
# Time-dependent PNP with quasi-static Poisson solve each RHS:
#   -εε0 ∇²φ = e (cp - cm)
# ============================================================

# --------------------------
# Geometry / grid
# --------------------------
const L = 5.0e-5
const a = 1.0e-5
const Nx = 160
const Ny = 40
const dx = L/(Nx-1)
const dy = a/(Ny-1)

# --------------------------
# Transport / constants
# --------------------------
const Dp = 1e-11
const Dm = 1e-11

const e  = 1.602176634e-19
const kB = 1.380649e-23
const T  = 273.15
const beta = e/(kB*T)                 # 1/V

const eps0 = 8.8541878128e-12
const epsr = 80.0                      # example (water-like); set as needed
const eps  = epsr * eps0

# Applied voltage
const phi0 = 0.0
const ΔV   = 1.0                       # volts
const phiL = ΔV

# Surface charge density on y-walls (C/m^2)
const qwall = 0.0                      # set nonzero to activate wall charge

# Gauss law: -ε ∂n φ = q  ->  ∂n φ = -q/ε  (outward normal derivative)
const dphi_dn_wall = -qwall/eps

# --------------------------
# Operators (LiquidCrystals)
# --------------------------
grad = LC.CenteredDifference{LC.Gradient}((dx,dy))
div  = LC.CenteredDifference{LC.Divergence}((dx,dy))

# We'll also use a Laplacian for applying BCs consistently, but Poisson will be solved by SOR.
lap = LC.CenteredDifference{LC.Laplacian}((dx,dy))

# --------------------------
# Boundary conditions (for applying ghost cells)
# --------------------------
# Species: Dirichlet in x, (we will override y ghosts to enforce NP no-flux)
bc_c = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(0.0, 0.0),
    LC.NeumannAxisBC{LC.YAxis}(0.0, 0.0),   # dummy; we'll overwrite y-ghosts for NP
))

# Potential: Dirichlet in x, Neumann in y based on wall charge
# Convention in earlier LC usage: NeumannAxisBC{YAxis}(low, high) corresponds to
# bottom and top outward normal derivatives with opposite sign handling.
# A safe consistent choice is:
#   bottom: ∂φ/∂y = -dphi_dn_wall   (since n=-ŷ)
#   top:    ∂φ/∂y = +dphi_dn_wall   (since n=+ŷ)
bc_phi = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(phi0, phiL),
    LC.NeumannAxisBC{LC.YAxis}(-dphi_dn_wall, +dphi_dn_wall),
))

# --------------------------
# Initial condition: c0 (Nx×Ny×2)
# --------------------------
c0 = ones(Float64, Nx, Ny, 2)          # cp=cm=1 everywhere initially (you can change)

# --------------------------
# Caches for concentrations and phi
# --------------------------
cp_ext, grad_cp_ext = LC.build_caches(grad, bc_c, view(c0,:,:,1))
cm_ext, grad_cm_ext = LC.build_caches(grad, bc_c, view(c0,:,:,2))

# Flux arrays and divergence buffers
Jp_ext = similar(grad_cp_ext)
Jm_ext = similar(grad_cm_ext)
divJp_ext = zeros(Float64, size(cp_ext)...)
divJm_ext = zeros(Float64, size(cm_ext)...)

# Potential arrays/caches for gradient
phi = zeros(Float64, Nx, Ny)
phi_ext, grad_phi_ext = LC.build_caches(grad, bc_phi, phi)

# --------------------------
# Helper: enforce NP no-flux for species on y-walls using wall normal E-field
# n·J = 0 => ∂n c ± beta*c*∂n phi = 0
# We implement via ghost values in the extended concentration arrays.
# --------------------------
function apply_np_y_noflux!(c_ext::AbstractMatrix, zsign::Float64, beta::Float64,
                            dphi_dn::Float64, dy::Float64)
    # c_ext has 1 ghost layer. Interior rows are 2:end-1.
    # bottom wall: outward normal n = -ŷ.  Our scalar dphi_dn is outward derivative ∂n φ.
    # For the BC we need outward form: ∂n c + zsign*beta*c*∂n φ = 0.
    #
    # Bottom ghost row j=1 adjacent interior j=2:
    #   ∂n c ≈ (c_g - c_b)/(2dy)   (outward at bottom points to -y, but this formula is outward in ghost direction)
    # We'll use the same ghost formula for both walls in terms of outward derivative:
    #   c_g = c_b + 2dy * ∂n c
    #
    # and ∂n c = - zsign*beta*c_b*dphi_dn.
    #
    # Top wall ghost row j=end adjacent interior j=end-1, outward +ŷ:
    #   c_g = c_b + 2dy * ∂n c   with b = end-1 and g = end.
    #
    @inbounds for i in 2:size(c_ext,1)-1
        # bottom
        cb = c_ext[i,2]
        c_ext[i,1] = cb + 2dy * (-(zsign*beta*cb*dphi_dn))
        # top
        ct = c_ext[i,end-1]
        c_ext[i,end] = ct + 2dy * (-(zsign*beta*ct*dphi_dn))
    end
    return nothing
end

# --------------------------
# Poisson solve: SOR on interior with BC enforcement each sweep
# -eps * Lap(phi) = e (cp - cm)
# --------------------------
function solve_poisson_SOR!(phi::AbstractMatrix, cp::AbstractMatrix, cm::AbstractMatrix,
                            bc_phi, phi_ext::AbstractMatrix, eps::Float64, echarge::Float64,
                            dx::Float64, dy::Float64;
                            ω::Float64=1.7, maxiter::Int=400, tol::Float64=1e-8)

    dx2 = dx*dx
    dy2 = dy*dy
    inv_denom = 1.0/(2.0/dx2 + 2.0/dy2)

    for it in 1:maxiter
        # enforce phi BCs into ghost cells so boundary behavior stays consistent
        LC.apply_BCs(lap, phi, bc_phi, phi_ext)

        maxupd = 0.0
        @inbounds for j in 2:Ny-1, i in 2:Nx-1
            rhs = -(echarge/eps) * (cp[i,j] - cm[i,j])  # Lap(phi) = rhs
            phi_new = ((phi[i+1,j] + phi[i-1,j])/dx2 + (phi[i,j+1] + phi[i,j-1])/dy2 - rhs) * inv_denom
            upd = ω*(phi_new - phi[i,j])
            phi[i,j] += upd
            maxupd = max(maxupd, abs(upd))
        end
        if maxupd < tol
            return it
        end
    end
    return maxiter
end

# --------------------------
# RHS: update phi from Poisson, then compute NP flux divergence
# --------------------------
function rhs_pnp!(du, u, p, t)
    (grad, div, bc_c,
     cp_ext, grad_cp_ext,
     cm_ext, grad_cm_ext,
     Jp_ext, Jm_ext,
     divJp_ext, divJm_ext,
     phi, bc_phi, phi_ext, grad_phi_ext,
     Dp, Dm, beta, eps, echarge, dx, dy, dphi_dn_wall) = p

    cp  = @view u[:,:,1];  dcp = @view du[:,:,1]
    cm  = @view u[:,:,2];  dcm = @view du[:,:,2]

    # --- Solve Poisson for phi given current cp,cm ---
    solve_poisson_SOR!(phi, cp, cm, bc_phi, phi_ext, eps, echarge, dx, dy; ω=1.7, maxiter=400, tol=1e-8)

    # Fill ghost cells for phi and compute grad(phi) on extended grid
    LC.apply_BCs(grad, phi, bc_phi, phi_ext)
    mul!(grad_phi_ext, grad, phi_ext)

    # --- Concentrations: ghost-fill in x, then enforce NP no-flux on y using wall charge ---
    LC.apply_BCs(grad, cp, bc_c, cp_ext)
    LC.apply_BCs(grad, cm, bc_c, cm_ext)
    apply_np_y_noflux!(cp_ext, +1.0, beta, dphi_dn_wall, dy)
    apply_np_y_noflux!(cm_ext, -1.0, beta, dphi_dn_wall, dy)

    # Gradients of c
    mul!(grad_cp_ext, grad, cp_ext)
    mul!(grad_cm_ext, grad, cm_ext)

    # Clear flux arrays (avoid stale boundary data)
    fill!(Jp_ext, zero(eltype(Jp_ext)))
    fill!(Jm_ext, zero(eltype(Jm_ext)))

    # Build fluxes on interior extended indices:
    # J+ = -Dp(∇cp + beta*cp*∇phi)
    # J- = -Dm(∇cm - beta*cm*∇phi)
    @inbounds for j in 2:size(cp_ext,2)-1, i in 2:size(cp_ext,1)-1
        cpi = cp_ext[i,j]
        cmi = cm_ext[i,j]
        gφ  = grad_phi_ext[i,j]
        Jp_ext[i,j] = -Dp * (grad_cp_ext[i,j] + beta * cpi * gφ)
        Jm_ext[i,j] = -Dm * (grad_cm_ext[i,j] - beta * cmi * gφ)
    end

    # Divergence
    fill!(divJp_ext, 0.0); fill!(divJm_ext, 0.0)
    mul!(divJp_ext, div, Jp_ext)
    mul!(divJm_ext, div, Jm_ext)

    @views dcp .= -divJp_ext[2:end-1, 2:end-1]
    @views dcm .= -divJm_ext[2:end-1, 2:end-1]
    return nothing
end

params = (grad, div, bc_c,
          cp_ext, grad_cp_ext,
          cm_ext, grad_cm_ext,
          Jp_ext, Jm_ext,
          divJp_ext, divJm_ext,
          phi, bc_phi, phi_ext, grad_phi_ext,
          Dp, Dm, beta, eps, e, dx, dy, dphi_dn_wall)

t_f = 5.0
Nt  = 200
prob = ODEProblem(rhs_pnp!, c0, (0.0, t_f), params)

# Explicit is OK for your small D; if you increase D or charge, you may need stiff integrator.
sol = solve(prob, Tsit5(); dtmax=2e-5, saveat=t_f/Nt, save_start=true, progress=true)

cp_final = sol.u[end][:,:,1]
cm_final = sol.u[end][:,:,2]
println("Final cp range: ", extrema(cp_final))
println("Final cm range: ", extrema(cm_final))
println("Final charge density range (cp-cm): ", extrema(cp_final .- cm_final))

# ------------------------------------------------
# Poisson wrapper for animation frames
# ------------------------------------------------
function solve_poisson_frame!(phi_frame::AbstractMatrix, cp::AbstractMatrix, cm::AbstractMatrix)
    # reuse the same BCs + caches as in the simulation
    solve_poisson_SOR!(phi_frame, cp, cm, bc_phi, phi_ext,
                       eps, e, dx, dy;
                       ω=1.7, maxiter=400, tol=1e-8)
    return nothing
end

#-----------------------------------------------------------------------------------
#   Movie
#-----------------------------------------------------------------------------------
using CairoMakie

# grids
x = range(0, L, length=Nx)
y = range(0, a, length=Ny)

# prealloc
phi_frame = similar(phi)  # Nx×Ny

# figure + axes
fig = Figure(resolution=(900, 900))

ax1 = Axis(fig[1,1], title="Charge density  ρ = c₊ - c₋")
ax2 = Axis(fig[2,1], title="Salt  s = c₊ + c₋")
ax3 = Axis(fig[3,1], title="Potential  ϕ(x,y)", xlabel="x")

linkxaxes!(ax1, ax2, ax3)

# OPTIONAL: keep physical box aspect ratio (recommended)
ax1.aspect = DataAspect()
ax2.aspect = DataAspect()
ax3.aspect = DataAspect()

# initial frame
cp0 = sol.u[1][:,:,1]
cm0 = sol.u[1][:,:,2]
ρ0 = cp0 .- cm0
s0 = cp0 .+ cm0
solve_poisson_frame!(phi_frame, cp0, cm0)

# IMPORTANT: pass Nx×Ny directly (no transpose)
hm1 = heatmap!(ax1, x, y, ρ0; colormap=:balance)
hm2 = heatmap!(ax2, x, y, s0; colormap=:viridis)
hm3 = heatmap!(ax3, x, y, phi_frame; colormap=:plasma)

Colorbar(fig[1,2], hm1)
Colorbar(fig[2,2], hm2)
Colorbar(fig[3,2], hm3)

record(fig, "PNP_animation.mp4", 1:length(sol.u); framerate=20) do k
    cp = sol.u[k][:,:,1]
    cm = sol.u[k][:,:,2]

    ρ = cp .- cm
    s = cp .+ cm
    solve_poisson_frame!(phi_frame, cp, cm)

    # Update Z-data (3rd argument), NOT x-data (1st argument)
    hm1[3][] = ρ
    hm2[3][] = s
    hm3[3][] = phi_frame

    ax1.title = "Charge density  ρ = c₊ - c₋    t = $(round(sol.t[k], digits=4)) s"
end