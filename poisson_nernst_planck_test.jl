using OrdinaryDiffEq
using LinearAlgebra
using StaticArrays
using Statistics
using DelimitedFiles
using Printf
import LiquidCrystals as LC
using CairoMakie

# ============================================================
# Full face-flux PNP in a LiquidCrystals.jl-compatible style
#
#   ∂c₊/∂t = -∇·J₊
#   ∂c₋/∂t = -∇·J₋
#
#   J₊ = -D₊ (∇c₊ + c₊ ∇ϕ)
#   J₋ = -D₋ (∇c₋ - c₋ ∇ϕ)
#
#   ∇²ϕ = -(κ²/2) (c₊ - c₋)
#
# Boundary conditions:
#   ϕ(0,y)=0, ϕ(L,y)=ϕL
#   ∂ϕ/∂n = -qwall_tilde at y-walls
#
#   c₊(0,y)=cL_p, c₊(L,y)=cR_p
#   c₋(0,y)=cL_m, c₋(L,y)=cR_m
#   J_y = 0 at top/bottom walls for both species
#
# Notes:
# - species are cell-centered
# - fluxes are face-centered
# - y-wall no-flux is imposed directly on the wall faces
# ============================================================

# --------------------------
# Physical scales
# --------------------------
const L_phys = 5.0e-5
const a_phys = 1.0e-5

const Dp_phys = 1e-11
const Dm_phys = 1e-11
const Dref    = Dp_phys

const echarge = 1.602176634e-19
const kB      = 1.380649e-23
const T       = 273.15

const eps0 = 8.8541878128e-12
const epsr = 80.0
const eps  = epsr * eps0

const ΔV_phys    = 0.01
const qwall_phys = 0.0

# number density
const c_res_phys = 6.023e23 * 0.01

# --------------------------
# Nondimensional parameters
# --------------------------
const Ltilde = L_phys / a_phys
const Dp = Dp_phys / Dref
const Dm = Dm_phys / Dref

const kappa2 = 2.0 * echarge^2 * a_phys^2 * c_res_phys / (eps * kB * T)
println("kappa2 = ", kappa2)

const phi0 = 0.0
const phiL = echarge * ΔV_phys / (kB * T)
println("phiL = ", phiL)

const qwall_tilde = echarge * a_phys * qwall_phys / (eps * kB * T)
println("qwall_tilde = ", qwall_tilde)

# concentration Dirichlet data
const cL_p = 1.0
const cR_p = 1.0
const cL_m = 1.0
const cR_m = 1.0

# --------------------------
# Geometry / grid
# --------------------------
const Nx = 160
const Ny = 40
const dx = Ltilde / (Nx - 1)
const dy = 1.0 / (Ny - 1)

const dphidy_bottom = qwall_tilde
const dphidy_top    = -qwall_tilde

# --------------------------
# LC operators / BC structs
# --------------------------
grad = LC.CenteredDifference{LC.Gradient}((dx, dy))
lap  = LC.CenteredDifference{LC.Laplacian}((dx, dy))

bc_phi = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(phi0, phiL),
    LC.NeumannAxisBC{LC.YAxis}(dphidy_bottom, dphidy_top),
))

# purely for consistency / metadata; species update itself is face-flux based
bc_cp = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(cL_p, cR_p),
    LC.NeumannAxisBC{LC.YAxis}(0.0, 0.0),
))
bc_cm = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(cL_m, cR_m),
    LC.NeumannAxisBC{LC.YAxis}(0.0, 0.0),
))

# --------------------------
# Initial condition
# --------------------------
c0 = zeros(Float64, Nx, Ny, 2)

# --------------------------
# Caches
# --------------------------
phi = zeros(Float64, Nx, Ny)
phi_ext, grad_phi_ext = LC.build_caches(grad, bc_phi, phi)

# face flux arrays
# x-faces: (Nx+1, Ny)
Jxp = zeros(Float64, Nx+1, Ny)
Jxm = zeros(Float64, Nx+1, Ny)

# y-faces: (Nx, Ny+1)
Jyp = zeros(Float64, Nx, Ny+1)
Jym = zeros(Float64, Nx, Ny+1)

# divergence arrays at cell centers
divJp = zeros(Float64, Nx, Ny)
divJm = zeros(Float64, Nx, Ny)

# --------------------------
# Poisson solve by Gauss-Seidel / SOR
# --------------------------
function solve_poisson_SOR!(
    phi::AbstractMatrix,
    cp::AbstractMatrix,
    cm::AbstractMatrix,
    phi0::Float64,
    phiL::Float64,
    dphidy_bottom::Float64,
    dphidy_top::Float64,
    kappa2::Float64,
    dx::Float64,
    dy::Float64;
    ω::Float64 = 1.1,
    maxiter::Int = 2000,
    tol::Float64 = 1e-6,
)
    Nx, Ny = size(phi)
    dx2 = dx * dx
    dy2 = dy * dy
    invden = 1.0 / (2.0 / dx2 + 2.0 / dy2)

    for it in 1:maxiter
        # x Dirichlet
        @inbounds for j in 1:Ny
            phi[1, j]  = phi0
            phi[Nx, j] = phiL
        end

        # y Neumann
        @inbounds for i in 2:Nx-1
            phi[i, 1]  = phi[i, 2]    - dy * dphidy_bottom
            phi[i, Ny] = phi[i, Ny-1] + dy * dphidy_top
        end

        # corners
        phi[1,1]   = phi0
        phi[1,Ny]  = phi0
        phi[Nx,1]  = phiL
        phi[Nx,Ny] = phiL

        maxupd = 0.0

        @inbounds for j in 2:Ny-1, i in 2:Nx-1
            rhs = -(kappa2 / 2.0) * (cp[i,j] - cm[i,j])

            phi_new = (
                (phi[i+1,j] + phi[i-1,j]) / dx2 +
                (phi[i,j+1] + phi[i,j-1]) / dy2 -
                rhs
            ) * invden

            upd = ω * (phi_new - phi[i,j])
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
# Full NP face flux divergence
# --------------------------
function compute_np_flux_divergence!(
    divJp::AbstractMatrix,
    divJm::AbstractMatrix,
    Jxp::AbstractMatrix,
    Jxm::AbstractMatrix,
    Jyp::AbstractMatrix,
    Jym::AbstractMatrix,
    cp::AbstractMatrix,
    cm::AbstractMatrix,
    phi::AbstractMatrix,
    Dp::Float64,
    Dm::Float64,
    dx::Float64,
    dy::Float64,
    cL_p::Float64, cR_p::Float64,
    cL_m::Float64, cR_m::Float64,
    phi0::Float64, phiL::Float64,
)
    Nx, Ny = size(cp)

    fill!(Jxp, 0.0)
    fill!(Jxm, 0.0)
    fill!(Jyp, 0.0)
    fill!(Jym, 0.0)
    fill!(divJp, 0.0)
    fill!(divJm, 0.0)

    # --------------------------------------------------
    # x-faces
    # face i = 1 is left boundary face
    # face i = Nx+1 is right boundary face
    # --------------------------------------------------

    # left boundary face: use prescribed boundary values
    @inbounds for j in 1:Ny
        dcpdx  = (cp[1,j] - cL_p) / dx
        dcmdx  = (cm[1,j] - cL_m) / dx
        dphidx = (phi[1,j] - phi0) / dx

        cface_p = 0.5 * (cp[1,j] + cL_p)
        cface_m = 0.5 * (cm[1,j] + cL_m)

        Jxp[1,j] = -Dp * (dcpdx + cface_p * dphidx)
        Jxm[1,j] = -Dm * (dcmdx - cface_m * dphidx)
    end

    # interior x-faces
    @inbounds for j in 1:Ny, i in 2:Nx
        dcpdx  = (cp[i,j]  - cp[i-1,j])  / dx
        dcmdx  = (cm[i,j]  - cm[i-1,j])  / dx
        dphidx = (phi[i,j] - phi[i-1,j]) / dx

        cface_p = 0.5 * (cp[i,j] + cp[i-1,j])
        cface_m = 0.5 * (cm[i,j] + cm[i-1,j])

        Jxp[i,j] = -Dp * (dcpdx + cface_p * dphidx)
        Jxm[i,j] = -Dm * (dcmdx - cface_m * dphidx)
    end

    # right boundary face
    @inbounds for j in 1:Ny
        dcpdx  = (cR_p - cp[Nx,j]) / dx
        dcmdx  = (cR_m - cm[Nx,j]) / dx
        dphidx = (phiL - phi[Nx,j]) / dx

        cface_p = 0.5 * (cp[Nx,j] + cR_p)
        cface_m = 0.5 * (cm[Nx,j] + cR_m)

        Jxp[Nx+1,j] = -Dp * (dcpdx + cface_p * dphidx)
        Jxm[Nx+1,j] = -Dm * (dcmdx - cface_m * dphidx)
    end

    # --------------------------------------------------
    # y-faces
    # wall faces are no-flux for TOTAL NP flux
    # --------------------------------------------------
    @inbounds for i in 1:Nx
        Jyp[i,1]    = 0.0
        Jyp[i,Ny+1] = 0.0
        Jym[i,1]    = 0.0
        Jym[i,Ny+1] = 0.0
    end

    # interior y-faces
    @inbounds for i in 1:Nx, j in 2:Ny
        dcpdy  = (cp[i,j]  - cp[i,j-1])  / dy
        dcmdy  = (cm[i,j]  - cm[i,j-1])  / dy
        dphidy = (phi[i,j] - phi[i,j-1]) / dy

        cface_p = 0.5 * (cp[i,j] + cp[i,j-1])
        cface_m = 0.5 * (cm[i,j] + cm[i,j-1])

        Jyp[i,j] = -Dp * (dcpdy + cface_p * dphidy)
        Jym[i,j] = -Dm * (dcmdy - cface_m * dphidy)
    end

    # --------------------------------------------------
    # divergence back to centers
    # --------------------------------------------------
    @inbounds for i in 1:Nx, j in 1:Ny
        divJp[i,j] = (Jxp[i+1,j] - Jxp[i,j]) / dx + (Jyp[i,j+1] - Jyp[i,j]) / dy
        divJm[i,j] = (Jxm[i+1,j] - Jxm[i,j]) / dx + (Jym[i,j+1] - Jym[i,j]) / dy
    end

    return nothing
end

# --------------------------
# RHS
# --------------------------
function rhs_pnp_faceflux_full!(du, u, p, t)
    (
        phi,
        Jxp, Jxm, Jyp, Jym,
        divJp, divJm,
        Dp, Dm, kappa2, dx, dy,
        dphidy_bottom, dphidy_top,
        cL_p, cR_p,
        cL_m, cR_m,
        phi0, phiL,
    ) = p

    cp  = @view u[:,:,1]
    cm  = @view u[:,:,2]
    dcp = @view du[:,:,1]
    dcm = @view du[:,:,2]

    # strongly enforce x-Dirichlet values in state
    @views begin
        cp[1,:]   .= cL_p
        cp[end,:] .= cR_p
        cm[1,:]   .= cL_m
        cm[end,:] .= cR_m
    end

    # quasi-static Poisson solve
    solve_poisson_SOR!(
        phi, cp, cm,
        phi0, phiL,
        dphidy_bottom, dphidy_top,
        kappa2, dx, dy;
        ω = 1.2,
        maxiter = 2000,
        tol = 1e-6
    )

    # full NP face flux divergence
    compute_np_flux_divergence!(
        divJp, divJm,
        Jxp, Jxm, Jyp, Jym,
        cp, cm, phi,
        Dp, Dm,
        dx, dy,
        cL_p, cR_p,
        cL_m, cR_m,
        phi0, phiL
    )

    @views begin
        dcp .= -divJp
        dcm .= -divJm
    end

    # freeze Dirichlet nodes in time
    @views begin
        dcp[1,:]   .= 0.0
        dcp[end,:] .= 0.0
        dcm[1,:]   .= 0.0
        dcm[end,:] .= 0.0
    end

    return nothing
end

params = (
    phi,
    Jxp, Jxm, Jyp, Jym,
    divJp, divJm,
    Dp, Dm, kappa2, dx, dy,
    dphidy_bottom, dphidy_top,
    cL_p, cR_p,
    cL_m, cR_m,
    phi0, phiL,
)

# --------------------------
# Solve
# --------------------------
t_f = 5
Nt  = 100

prob = ODEProblem(rhs_pnp_faceflux_full!, c0, (0.0, t_f), params)

sol = solve(
    prob,
    Tsit5();
    dtmax = 2e-5,
    saveat = t_f / Nt,
    save_start = true,
    progress = true
)

cp_final = sol.u[end][:,:,1]
cm_final = sol.u[end][:,:,2]

println("Final cp range: ", extrema(cp_final))
println("Final cm range: ", extrema(cm_final))
println("Final charge density range (cp-cm): ", extrema(cp_final .- cm_final))

##
# --------------------------------------------------------------------------
# Reconstruct phi history
# --------------------------------------------------------------------------
function reconstruct_phi_history(
    sol,
    Nx, Ny,
    phi0, phiL,
    dphidy_bottom, dphidy_top,
    kappa2, dx, dy
)
    phi_hist = Matrix{Float64}[]

    for n in 1:length(sol.u)
        u  = sol.u[n]
        cp = Array(@view u[:,:,1])
        cm = Array(@view u[:,:,2])

        phi_n = zeros(Float64, Nx, Ny)

        solve_poisson_SOR!(
            phi_n, cp, cm,
            phi0, phiL,
            dphidy_bottom, dphidy_top,
            kappa2, dx, dy;
            ω = 1.0,
            maxiter = 4000,
            tol = 1e-8
        )

        push!(phi_hist, phi_n)
    end

    return phi_hist
end


# --------------------------------------------------------------------------
# Scalar diagnostics
# --------------------------------------------------------------------------
function save_scalar_diagnostics(
    filename::String,
    sol,
    phi_hist::Vector{Matrix{Float64}},
    dx::Float64,
    dy::Float64
)
    Nt = length(sol.u)

    header = [
        "t",
        "cp_min", "cp_max", "cm_min", "cm_max",
        "phi_min", "phi_max",
        "rho_min", "rho_max",
        "cp_mean", "cm_mean", "rho_mean",
        "cp_mass", "cm_mass",
        "rho_l2"
    ]

    data = zeros(Float64, Nt, length(header))

    for n in 1:Nt
        u   = sol.u[n]
        cp  = @view u[:,:,1]
        cm  = @view u[:,:,2]
        phi = phi_hist[n]
        rho = cp .- cm

        data[n,1]  = sol.t[n]
        data[n,2]  = minimum(cp)
        data[n,3]  = maximum(cp)
        data[n,4]  = minimum(cm)
        data[n,5]  = maximum(cm)
        data[n,6]  = minimum(phi)
        data[n,7]  = maximum(phi)
        data[n,8]  = minimum(rho)
        data[n,9]  = maximum(rho)
        data[n,10] = mean(cp)
        data[n,11] = mean(cm)
        data[n,12] = mean(rho)
        data[n,13] = sum(cp) * dx * dy
        data[n,14] = sum(cm) * dx * dy
        data[n,15] = sqrt(sum(abs2, rho) * dx * dy)
    end

    open(filename, "w") do io
        writedlm(io, permutedims(header), ',')
        writedlm(io, data, ',')
    end
end


# --------------------------------------------------------------------------
# Save full field snapshots
# --------------------------------------------------------------------------
function save_field_snapshots(
    outdir::String,
    sol,
    phi_hist::Vector{Matrix{Float64}}
)
    mkpath(outdir)

    for n in 1:length(sol.u)
        u   = sol.u[n]
        cp  = Array(@view u[:,:,1])
        cm  = Array(@view u[:,:,2])
        phi = phi_hist[n]
        rho = cp .- cm

        tag = @sprintf("%04d", n)

        writedlm(joinpath(outdir, "cp_$tag.txt"),  cp)
        writedlm(joinpath(outdir, "cm_$tag.txt"),  cm)
        writedlm(joinpath(outdir, "phi_$tag.txt"), phi)
        writedlm(joinpath(outdir, "rho_$tag.txt"), rho)
    end
end


# --------------------------------------------------------------------------
# Movie
# --------------------------------------------------------------------------
function make_pnp_movie(
    filename::String,
    sol,
    phi_hist::Vector{Matrix{Float64}},
    Ltilde::Float64;
    framerate::Int = 15
)
    Nx, Ny, _ = size(sol.u[1])

    x = range(0, Ltilde, length=Nx)
    y = range(0, 1.0, length=Ny)

    cp0   = Array(@view sol.u[1][:,:,1])
    cm0   = Array(@view sol.u[1][:,:,2])
    phi00 = phi_hist[1]

    cp_min  = minimum(minimum(@view u[:,:,1]) for u in sol.u)
    cp_max  = maximum(maximum(@view u[:,:,1]) for u in sol.u)
    cm_min  = minimum(minimum(@view u[:,:,2]) for u in sol.u)
    cm_max  = maximum(maximum(@view u[:,:,2]) for u in sol.u)
    phi_min = minimum(minimum(φ) for φ in phi_hist)
    phi_max = maximum(maximum(φ) for φ in phi_hist)

    fig = Figure(resolution = (1800, 520))

    ax1 = Axis(fig[1,1], title = "c₊", xlabel = "x̃", ylabel = "ỹ", aspect = DataAspect())
    ax2 = Axis(fig[1,2], title = "c₋", xlabel = "x̃", ylabel = "ỹ", aspect = DataAspect())
    ax3 = Axis(fig[1,3], title = "ϕ",  xlabel = "x̃", ylabel = "ỹ", aspect = DataAspect())

    cp_obs  = Observable(cp0)
    cm_obs  = Observable(cm0)
    phi_obs = Observable(phi00)

    hm1 = heatmap!(ax1, x, y, cp_obs;  colorrange = (cp_min, cp_max))
    hm2 = heatmap!(ax2, x, y, cm_obs;  colorrange = (cm_min, cm_max))
    hm3 = heatmap!(ax3, x, y, phi_obs; colorrange = (phi_min, phi_max))

    Colorbar(fig[2,1], hm1, vertical=false)
    Colorbar(fig[2,2], hm2, vertical=false)
    Colorbar(fig[2,3], hm3, vertical=false)

    ttl = Label(fig[0,1:3], "t = $(round(sol.t[1], sigdigits=5))", fontsize = 22)

    record(fig, filename, 1:length(sol.u); framerate=framerate) do n
        cp_obs[]  = Array(@view sol.u[n][:,:,1])
        cm_obs[]  = Array(@view sol.u[n][:,:,2])
        phi_obs[] = phi_hist[n]
        ttl.text = "t = $(round(sol.t[n], sigdigits=5))"
    end
end


# --------------------------------------------------------------------------
# 1D mean and spread diagnostic
# --------------------------------------------------------------------------
function plot_1d_mean_and_spread(
    sol,
    phi_hist,
    Ltilde;
    filename="profiles_mean_spread.png"
)
    Nx, Ny, _ = size(sol.u[1])
    x = range(0, Ltilde, length=Nx)

    i0  = 1
    im  = Int(cld(length(sol.u), 2))
    ifn = length(sol.u)
    inds = [i0, im, ifn]

    fig = Figure(resolution=(1200, 1200))

    ax1 = Axis(fig[1,1], title="mean(c₊) vs x", xlabel="x̃", ylabel="⟨c₊⟩y")
    ax2 = Axis(fig[2,1], title="spread(c₊) vs x", xlabel="x̃", ylabel="max_y-min_y")
    ax3 = Axis(fig[3,1], title="mean(c₋) vs x", xlabel="x̃", ylabel="⟨c₋⟩y")
    ax4 = Axis(fig[4,1], title="spread(c₋) vs x", xlabel="x̃", ylabel="max_y-min_y")
    ax5 = Axis(fig[5,1], title="mean(ϕ) vs x",  xlabel="x̃", ylabel="⟨ϕ⟩y")
    ax6 = Axis(fig[6,1], title="spread(ϕ) vs x", xlabel="x̃", ylabel="max_y-min_y")

    for n in inds
        lab = "t = $(round(sol.t[n], sigdigits=4))"

        cp  = Array(@view sol.u[n][:,:,1])
        cm  = Array(@view sol.u[n][:,:,2])
        phi = phi_hist[n]

        cp_mean   = vec(mean(cp, dims=2))
        cm_mean   = vec(mean(cm, dims=2))
        phi_mean  = vec(mean(phi, dims=2))

        cp_spread   = vec(maximum(cp, dims=2) .- minimum(cp, dims=2))
        cm_spread   = vec(maximum(cm, dims=2) .- minimum(cm, dims=2))
        phi_spread  = vec(maximum(phi, dims=2) .- minimum(phi, dims=2))

        lines!(ax1, x, cp_mean, label=lab)
        lines!(ax2, x, cp_spread)
        lines!(ax3, x, cm_mean, label=lab)
        lines!(ax4, x, cm_spread)
        lines!(ax5, x, phi_mean, label=lab)
        lines!(ax6, x, phi_spread)
    end

    axislegend(ax1, position=:rb)
    save(filename, fig)
    return fig
end


# --------------------------------------------------------------------------
# Wall flux diagnostic consistent with face-flux formulation
# --------------------------------------------------------------------------
function save_wall_fluxes_faceflux(
    outdir::String,
    sol,
    phi_hist,
    Dp::Float64,
    Dm::Float64,
    dx::Float64,
    dy::Float64
)
    mkpath(outdir)

    for n in 1:length(sol.u)
        u   = sol.u[n]
        cp  = Array(@view u[:,:,1])
        cm  = Array(@view u[:,:,2])
        phi = phi_hist[n]

        # total y-flux at bottom/top walls using one-sided diffusive derivative
        # + migration contribution
        bottom_flux_p = zeros(Float64, Nx)
        bottom_flux_m = zeros(Float64, Nx)
        top_flux_p    = zeros(Float64, Nx)
        top_flux_m    = zeros(Float64, Nx)

        @inbounds for i in 1:Nx
            # bottom wall
            dcpdy_b  = (cp[i,2]  - cp[i,1])  / dy
            dcmdy_b  = (cm[i,2]  - cm[i,1])  / dy
            dphidy_b = (phi[i,2] - phi[i,1]) / dy

            bottom_flux_p[i] = -Dp * (dcpdy_b + cp[i,1] * dphidy_b)
            bottom_flux_m[i] = -Dm * (dcmdy_b - cm[i,1] * dphidy_b)

            # top wall
            dcpdy_t  = (cp[i,Ny]  - cp[i,Ny-1])  / dy
            dcmdy_t  = (cm[i,Ny]  - cm[i,Ny-1])  / dy
            dphidy_t = (phi[i,Ny] - phi[i,Ny-1]) / dy

            top_flux_p[i] = -Dp * (dcpdy_t + cp[i,Ny] * dphidy_t)
            top_flux_m[i] = -Dm * (dcmdy_t - cm[i,Ny] * dphidy_t)
        end

        tag = lpad(n, 4, "0")

        writedlm(joinpath(outdir, "bottom_flux_p_$tag.txt"), bottom_flux_p)
        writedlm(joinpath(outdir, "bottom_flux_m_$tag.txt"), bottom_flux_m)
        writedlm(joinpath(outdir, "top_flux_p_$tag.txt"),    top_flux_p)
        writedlm(joinpath(outdir, "top_flux_m_$tag.txt"),    top_flux_m)
    end
end

function plot_odd_even_check(sol, Ltilde; filename="odd_even_check.png")

    Nx, Ny, _ = size(sol.u[1])
    x = range(0, Ltilde, length=Nx)

    # choose final timestep
    cp = Array(@view sol.u[end][:,:,1])
    cm = Array(@view sol.u[end][:,:,2])

    # y-average
    cp_mean = vec(mean(cp, dims=2))
    cm_mean = vec(mean(cm, dims=2))

    odd  = 1:2:Nx
    even = 2:2:Nx

    fig = Figure(resolution=(900,600))

    ax1 = Axis(fig[1,1], title="Odd–even test for c₊", xlabel="x̃", ylabel="c₊")
    scatter!(ax1, x[odd],  cp_mean[odd],  label="odd nodes")
    scatter!(ax1, x[even], cp_mean[even], label="even nodes")
    axislegend(ax1)

    ax2 = Axis(fig[2,1], title="Odd–even test for c₋", xlabel="x̃", ylabel="c₋")
    scatter!(ax2, x[odd],  cm_mean[odd])
    scatter!(ax2, x[even], cm_mean[even])

    save(filename, fig)

    return fig
end


##
# -------------------------------------------------------------
# Reconstruct phi history
# -------------------------------------------------------------
phi_hist = reconstruct_phi_history(
    sol,
    Nx, Ny,
    phi0, phiL,
    dphidy_bottom, dphidy_top,
    kappa2,
    dx, dy
)

# -------------------------------------------------------------
# Scalar diagnostics (time evolution)
# -------------------------------------------------------------
save_scalar_diagnostics(
    "scalar_diagnostics.csv",
    sol,
    phi_hist,
    dx, dy
)

# -------------------------------------------------------------
# Save full field snapshots (cp, cm, phi, rho)
# -------------------------------------------------------------
save_field_snapshots(
    "field_snapshots",
    sol,
    phi_hist
)

# -------------------------------------------------------------
# 1D diagnostics (mean + spread vs x)
# -------------------------------------------------------------
plot_1d_mean_and_spread(
    sol,
    phi_hist,
    Ltilde;
    filename = "diagnostics_profiles.png"
)

# -------------------------------------------------------------
# Odd–even checkerboard test
# -------------------------------------------------------------
plot_odd_even_check(
    sol,
    Ltilde;
    filename = "odd_even_check.png"
)

# -------------------------------------------------------------
# Wall flux diagnostics (face-flux version)
# -------------------------------------------------------------
save_wall_fluxes_faceflux(
    "wall_flux",
    sol,
    phi_hist,
    Dp, Dm,
    dx, dy
)

# -------------------------------------------------------------
# Movie of cp, cm, phi evolution
# -------------------------------------------------------------
make_pnp_movie(
    "pnp_fields.mp4",
    sol,
    phi_hist,
    Ltilde;
    framerate = 15
)