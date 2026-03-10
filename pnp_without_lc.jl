using OrdinaryDiffEq
using LinearAlgebra
using StaticArrays
import LiquidCrystals as LC
using CairoMakie

# ============================================================
# Nondimensional PNP with LiquidCrystals.jl formalism
#
#   ∂c₊/∂t = -∇·J₊,   J₊ = -D₊ (∇c₊ + c₊ ∇ϕ)
#   ∂c₋/∂t = -∇·J₋,   J₋ = -D₋ (∇c₋ - c₋ ∇ϕ)
#
#   ∇²ϕ = -(κ²/2) (c₊ - c₋)
#
# BCs:
#   ϕ(0,y)=0,  ϕ(L,y)=ΔV
#   ∂ϕ/∂n = -qwall_tilde on y-walls
#
#   c₊(0,y)=c₊(L,y)=1
#   c₋(0,y)=c₋(L,y)=1
#   ∂n c± ± c± ∂nϕ = 0 on y-walls
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

# realistic dimensional reservoir concentration (#/m^3)
const c_res_phys = 6.023e23 * 0.01

# --------------------------
# Nondimensional parameters
# --------------------------
const Ltilde = L_phys / a_phys
const Dp = Dp_phys / Dref
const Dm = Dm_phys / Dref

# κ² = 2 e² a² c_res / (ε kB T)
const kappa2 = 2.0 * echarge^2 * a_phys^2 * c_res_phys / (eps * kB * T)
println(kappa2)

# ΔṼ = e ΔV / (kB T)
const phi0 = 0.0
const phiL = echarge * ΔV_phys / (kB * T)
println(phiL)

# q̃wall = e a qwall / (ε kB T)
const qwall_tilde = echarge * a_phys * qwall_phys / (eps * kB * T)
println(qwall_tilde)

# reservoir concentrations are 1 in nondimensional variables
const c_res = 1.0

# --------------------------
# Geometry / grid
# --------------------------
const Nx = 160
const Ny = 40
const dx = Ltilde / (Nx - 1)
const dy = 1.0 / (Ny - 1)

# --------------------------
# LiquidCrystals operators
# --------------------------
grad = LC.CenteredDifference{LC.Gradient}((dx, dy))
div  = LC.CenteredDifference{LC.Divergence}((dx, dy))
lap  = LC.CenteredDifference{LC.Laplacian}((dx, dy))

# --------------------------
# Boundary conditions for ghost-cell handling
# --------------------------
# Species: Dirichlet in x, dummy Neumann in y (we overwrite y ghosts manually)
bc_c = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(c_res, 0.0),
    LC.NeumannAxisBC{LC.YAxis}(0.0, 0.0),
))

# Potential: Dirichlet in x, Neumann in y from wall charge
# bottom: ∂ϕ/∂y = +qwall_tilde
# top:    ∂ϕ/∂y = -qwall_tilde
const dphidy_bottom =  qwall_tilde
const dphidy_top    = -qwall_tilde

bc_phi = LC.BoxBC((
    LC.DirichletAxisBC{LC.XAxis}(phi0, phiL),
    LC.NeumannAxisBC{LC.YAxis}(dphidy_bottom, dphidy_top),
))

# --------------------------
# Initial condition
# --------------------------
c0 = zeros(Float64, Nx, Ny, 2)

# --------------------------
# Caches
# --------------------------
cp_ext, grad_cp_ext = LC.build_caches(grad, bc_c, view(c0, :, :, 1))
cm_ext, grad_cm_ext = LC.build_caches(grad, bc_c, view(c0, :, :, 2))

Jp_ext = similar(grad_cp_ext)
Jm_ext = similar(grad_cm_ext)

divJp_ext = zeros(Float64, size(cp_ext)...)
divJm_ext = zeros(Float64, size(cm_ext)...)

phi = zeros(Float64, Nx, Ny)
phi_ext, grad_phi_ext = LC.build_caches(grad, bc_phi, phi)

# --------------------------
# Enforce NP no-flux on y-walls via species ghost values
# ∂n c± ± c± ∂nϕ = 0
# --------------------------
function apply_np_y_noflux!(
    c_ext::AbstractMatrix,
    zsign::Float64,              # +1 for c+, -1 for c-
    dphidy_bottom::Float64,
    dphidy_top::Float64,
    dy::Float64
)
    @inbounds for i in 2:size(c_ext,1)-1
        cb = c_ext[i,2]
        ct = c_ext[i,end-1]

        # bottom wall ghost
        c_ext[i,1] = cb - dy * zsign * cb * dphidy_bottom

        # top wall ghost
        c_ext[i,end] = ct - dy * zsign * ct * dphidy_top
    end
    return nothing
end

# --------------------------
# Poisson solve by Gauss–Seidel / SOR
# ∇²ϕ = -(κ²/2)(c₊ - c₋)
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
    ω::Float64 = 1.7,
    maxiter::Int = 4000,
    tol::Float64 = 1e-8,
)
    Nx, Ny = size(phi)
    dx2 = dx * dx
    dy2 = dy * dy
    invden = 1.0 / (2.0/dx2 + 2.0/dy2)

    for it in 1:maxiter
        # x-Dirichlet
        @inbounds for j in 1:Ny
            phi[1,j]  = phi0
            phi[Nx,j] = phiL
        end

        # y-Neumann
        @inbounds for i in 2:Nx-1
            phi[i,1]  = phi[i,2]    - dy * dphidy_bottom
            phi[i,Ny] = phi[i,Ny-1] + dy * dphidy_top
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
# RHS: centered-flux formulation using LC structs/caches
# --------------------------
function rhs_pnp!(du, u, p, t)
    (
        grad, div, bc_c,
        cp_ext, grad_cp_ext,
        cm_ext, grad_cm_ext,
        Jp_ext, Jm_ext,
        divJp_ext, divJm_ext,
        phi, bc_phi, phi_ext, grad_phi_ext,
        Dp, Dm, kappa2, dx, dy,
        dphidy_bottom, dphidy_top
    ) = p

    cp  = @view u[:,:,1]
    cm  = @view u[:,:,2]
    dcp = @view du[:,:,1]
    dcm = @view du[:,:,2]

    # 1. solve Poisson quasi-statically
    solve_poisson_SOR!(
        phi, cp, cm,
        phi0, phiL,
        dphidy_bottom, dphidy_top,
        kappa2, dx, dy;
        ω = 1.0, maxiter = 4000, tol = 1e-8
    )

    # 2. apply potential BCs and compute grad(phi)
    LC.apply_BCs(grad, phi, bc_phi, phi_ext)
    mul!(grad_phi_ext, grad, phi_ext)

    # 3. species ghost cells
    LC.apply_BCs(grad, cp, bc_c, cp_ext)
    LC.apply_BCs(grad, cm, bc_c, cm_ext)

    # overwrite y ghosts with NP no-flux
    
    #apply_np_y_noflux!(cp_ext, +1.0, dphidy_bottom, dphidy_top, dy)
    #apply_np_y_noflux!(cm_ext, -1.0, dphidy_bottom, dphidy_top, dy)

    # 4. compute grad(c)
    mul!(grad_cp_ext, grad, cp_ext)
    mul!(grad_cm_ext, grad, cm_ext)

    # 5. build centered fluxes at grid points
    fill!(Jp_ext, zero(eltype(Jp_ext)))
    fill!(Jm_ext, zero(eltype(Jm_ext)))

    @inbounds for j in 2:size(cp_ext,2)-1, i in 2:size(cp_ext,1)-1
        cpi = cp_ext[i,j]
        cmi = cm_ext[i,j]
        gϕ  = grad_phi_ext[i,j]

        Jp_ext[i,j] = -Dp * (grad_cp_ext[i,j])# + cpi * gϕ)
        Jm_ext[i,j] = -Dm * (grad_cm_ext[i,j])# - cmi * gϕ)
    end

    # 6. centered divergence of flux
    fill!(divJp_ext, 0.0)
    fill!(divJm_ext, 0.0)

    mul!(divJp_ext, div, Jp_ext)
    mul!(divJm_ext, div, Jm_ext)

    @views dcp .= -divJp_ext[2:end-1, 2:end-1]
    @views dcm .= -divJm_ext[2:end-1, 2:end-1]

    return nothing
end

params = (
    grad, div, bc_c,
    cp_ext, grad_cp_ext,
    cm_ext, grad_cm_ext,
    Jp_ext, Jm_ext,
    divJp_ext, divJm_ext,
    phi, bc_phi, phi_ext, grad_phi_ext,
    Dp, Dm, kappa2, dx, dy,
    dphidy_bottom, dphidy_top
)

# --------------------------
# Solve
# --------------------------
t_f = 5.0
Nt  = 200

prob = ODEProblem(rhs_pnp!, c0, (0.0, t_f), params)

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
 

#--------------------------------------------------------------------------
#   Reconstructing phi
#--------------------------------------------------------------------------
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

        phi = zeros(Float64, Nx, Ny)

        solve_poisson_SOR!(
            phi, cp, cm,
            phi0, phiL,
            dphidy_bottom, dphidy_top,
            kappa2, dx, dy;
            ω = 1.0,
            maxiter = 4000,
            tol = 1e-8
        )

        push!(phi_hist, copy(phi))
    end

    return phi_hist
end

#--------------------------------------------------------------------------
#       Diagnostics
#--------------------------------------------------------------------------
using DelimitedFiles
using LinearAlgebra
using Statistics

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

using Printf

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

#----------------------------------------------------------------------------------------
#   Movie
#----------------------------------------------------------------------------------------
function make_pnp_movie(
    filename::String,
    sol,
    phi_hist::Vector{Matrix{Float64}},
    Ltilde::Float64;
    framerate::Int = 15
)
    Nx, Ny, _ = size(sol.u[1])

    x = range(0, Ltilde, length=Nx)
    y = range(0, 1.0,    length=Ny)

    cp0   = Array(@view sol.u[1][:,:,1])
    cm0   = Array(@view sol.u[1][:,:,2])
    phi00 = phi_hist[1]

    cp_min  = minimum([minimum(@view u[:,:,1]) for u in sol.u])
    cp_max  = maximum([maximum(@view u[:,:,1]) for u in sol.u])
    cm_min  = minimum([minimum(@view u[:,:,2]) for u in sol.u])
    cm_max  = maximum([maximum(@view u[:,:,2]) for u in sol.u])
    phi_min = minimum([minimum(φ) for φ in phi_hist])
    phi_max = maximum([maximum(φ) for φ in phi_hist])

    fig = Figure(resolution = (1800, 520))

    ax1 = Axis(fig[1,1], title = "c₊", xlabel = "x", ylabel = "y", aspect = DataAspect())
    ax2 = Axis(fig[1,2], title = "c₋", xlabel = "x", ylabel = "y", aspect = DataAspect())
    ax3 = Axis(fig[1,3], title = "ϕ",  xlabel = "x", ylabel = "y", aspect = DataAspect())

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

    record(fig, filename, 1:length(sol.u); framerate = framerate) do n
        cp_obs[]  = Array(@view sol.u[n][:,:,1])
        cm_obs[]  = Array(@view sol.u[n][:,:,2])
        phi_obs[] = phi_hist[n]
        ttl.text = "t = $(round(sol.t[n], sigdigits=5))"
    end
end

function save_wall_fluxes(
    outdir::String,
    sol,
    phi_hist,
    grad,
    bc_c,
    cp_ext, grad_cp_ext,
    cm_ext, grad_cm_ext,
    grad_phi_ext,
    Dp, Dm,
    dphidy_bottom,
    dphidy_top,
    dy
)

    mkpath(outdir)

    for n in 1:length(sol.u)

        u   = sol.u[n]
        cp  = @view u[:,:,1]
        cm  = @view u[:,:,2]
        phi = phi_hist[n]

        # rebuild ghost cells
        LC.apply_BCs(grad, cp, bc_c, cp_ext)
        LC.apply_BCs(grad, cm, bc_c, cm_ext)

        apply_np_y_noflux!(cp_ext, +1.0, dphidy_bottom, dphidy_top, dy)
        apply_np_y_noflux!(cm_ext, -1.0, dphidy_bottom, dphidy_top, dy)

        # rebuild gradients
        mul!(grad_cp_ext, grad, cp_ext)
        mul!(grad_cm_ext, grad, cm_ext)

        # rebuild potential gradient
        LC.apply_BCs(grad, phi, bc_phi, phi_ext)
        mul!(grad_phi_ext, grad, phi_ext)

        Nx_ext, Ny_ext = size(cp_ext)

        bottom_flux_p = Float64[]
        bottom_flux_m = Float64[]
        top_flux_p    = Float64[]
        top_flux_m    = Float64[]

        for i in 2:Nx_ext-1

            # bottom interior cell
            cpi = cp_ext[i,2]
            cmi = cm_ext[i,2]

            gcp = grad_cp_ext[i,2]
            gcm = grad_cm_ext[i,2]
            gϕ  = grad_phi_ext[i,2]

            Jp = -Dp * (gcp + cpi*gϕ)
            Jm = -Dm * (gcm - cmi*gϕ)

            push!(bottom_flux_p, Jp[2])
            push!(bottom_flux_m, Jm[2])

            # top interior cell
            cpi = cp_ext[i,Ny_ext-1]
            cmi = cm_ext[i,Ny_ext-1]

            gcp = grad_cp_ext[i,Ny_ext-1]
            gcm = grad_cm_ext[i,Ny_ext-1]
            gϕ  = grad_phi_ext[i,Ny_ext-1]

            Jp = -Dp * (gcp + cpi*gϕ)
            Jm = -Dm * (gcm - cmi*gϕ)

            push!(top_flux_p, Jp[2])
            push!(top_flux_m, Jm[2])

        end

        tag = lpad(n,4,"0")

        writedlm(joinpath(outdir,"bottom_flux_p_$tag.txt"), bottom_flux_p)
        writedlm(joinpath(outdir,"bottom_flux_m_$tag.txt"), bottom_flux_m)

        writedlm(joinpath(outdir,"top_flux_p_$tag.txt"), top_flux_p)
        writedlm(joinpath(outdir,"top_flux_m_$tag.txt"), top_flux_m)

    end
end

function plot_1d_mean_and_spread(sol, phi_hist, Ltilde; filename="profiles_mean_spread.png")
    Nx, Ny, _ = size(sol.u[1])
    x = range(0, Ltilde, length=Nx)

    i0 = 1
    im = Int(cld(length(sol.u), 2))
    ifn = length(sol.u)
    inds = [i0, im, ifn]

    fig = Figure(resolution=(1200, 1200))

    ax1 = Axis(fig[1,1], title="mean(c₊) vs x", xlabel="x̃", ylabel="⟨c₊⟩y")
    ax2 = Axis(fig[2,1], title="spread(c₊) vs x", xlabel="x̃", ylabel="max_y-min_y")
    ax3 = Axis(fig[3,1], title="mean(c₋) vs x", xlabel="x̃", ylabel="⟨c₋⟩y")
    ax4 = Axis(fig[4,1], title="spread(c₋) vs x", xlabel="x̃", ylabel="max_y-min_y")
    ax5 = Axis(fig[5,1], title="mean(ϕ) vs x", xlabel="x̃", ylabel="⟨ϕ⟩y")
    ax6 = Axis(fig[6,1], title="spread(ϕ) vs x", xlabel="x̃", ylabel="max_y-min_y")

    for n in inds
        lab = "t = $(round(sol.t[n], sigdigits=4))"

        cp  = Array(@view sol.u[n][:,:,1])
        cm  = Array(@view sol.u[n][:,:,2])
        phi = phi_hist[n]

        cp_mean = vec(mean(cp, dims=2))
        cm_mean = vec(mean(cm, dims=2))
        phi_mean = vec(mean(phi, dims=2))

        cp_spread = vec(maximum(cp, dims=2) .- minimum(cp, dims=2))
        cm_spread = vec(maximum(cm, dims=2) .- minimum(cm, dims=2))
        phi_spread = vec(maximum(phi, dims=2) .- minimum(phi, dims=2))

        lines!(ax1, x, cp_mean, label=lab)
        lines!(ax2, x, cp_spread, label=lab)
        lines!(ax3, x, cm_mean, label=lab)
        lines!(ax4, x, cm_spread, label=lab)
        lines!(ax5, x, phi_mean, label=lab)
        lines!(ax6, x, phi_spread, label=lab)
    end

    axislegend(ax1, position=:rb)
    save(filename, fig)
    return fig
end

#----------------------------------------------------------------------------------
#   Generate File
#-------------------------------------------------------------------------------------
phi_hist = reconstruct_phi_history(
    sol, Nx, Ny,
    phi0, phiL,
    dphidy_bottom, dphidy_top,
    kappa2, dx, dy
)

#save_scalar_diagnostics("scalar_diagnostics.csv", sol, phi_hist, dx, dy)
#save_field_snapshots("field_snapshots", sol, phi_hist)

make_pnp_movie("pnp_fields.mp4", sol, phi_hist, Ltilde; framerate=15)
plot_1d_mean_and_spread(sol, phi_hist, Ltilde, filename="diagnostics_profiles.png")
"""
save_wall_fluxes(
    "wall_flux",
    sol,
    phi_hist,
    grad,
    bc_c,
    cp_ext, grad_cp_ext,
    cm_ext, grad_cm_ext,
    grad_phi_ext,
    Dp, Dm,
    dphidy_bottom,
    dphidy_top,
    dy
)
"""

