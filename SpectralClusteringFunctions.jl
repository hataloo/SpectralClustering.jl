function cut(S, Sᶜ, W)
    return sum([W[i,j] for i in S, j in Sᶜ])
end

function vol(S, V, W)
    return sum([W[i,j] for i in S, j in V])
end

function Ncut(S, Sᶜ, W, V)
    return cut(S, Sᶜ, W)/vol(S,V,W) + cut(S, Sᶜ, W)/vol(Sᶜ, V, W)
end

function minNcut(ϕ₂, W, V)
    bestCut = 1e10
    τ = 0
    S = []
    Sᶜ = []
    for i = 1:size(ϕ₂,1)
        τᵢ = ϕ₂[i]
        Sᵢ = V[ϕ₂ .<= τᵢ]
        Sᶜᵢ = V[ϕ₂ .> τᵢ]
        currentCut = Ncut(Sᵢ, Sᶜᵢ, W, V)
        if  currentCut < bestCut
            bestCut = currentCut
            τ = τᵢ
            S = Sᵢ
            Sᶜ = Sᶜᵢ
        end
    end
    return τ, S, Sᶜ
end

function getSpectralUtility(V, W)
    d = [sum(W[i,:]) for i in V]
    D = diagm(d)
    D_reci_sqrt = inv(sqrt(D))
    L = D - W
    L_G = D_reci_sqrt * L * D_reci_sqrt
    ϕ = D_reci_sqrt * eigvecs(L_G)
    return D, L, L_G, ϕ
end

function SpectralClustering(k::Integer, V, W)
    _, _, _, ϕ =  getSpectralUtility(V, W)
    clustering = kmeans(transpose(ϕ[:, 2:k]), k)
    cluster_groups = assignments(clustering)
    clusters = [V[cluster_groups .== i] for i in 1:k]
    return clusters
end

function getNormals(N::Integer, R::Float64 = 1.0, sd_radial::Union{Float64,Vector{Float64}} = 1.0, sd_angular::Union{Float64,Vector{Float64}} = 1.0)
    sds = []
    for sd in [sd_radial, sd_angular]
        if sd isa Vector{Float64}
            @assert length(sd) == N
            push!(sds, sd)
        else 
            push!(sds, [sd for _ in 1:N])
        end
    end
    sd_radial, sd_angular = sds
    normals = []
    for i in 1:N
        θ = 2*π/(N) * (i-1)
        μ = [R * cos(θ), R * sin(θ)]
        p_1 = [cos(θ), sin(θ)]
        p_2 = [-sin(θ), cos(θ)]
        P = hcat(p_1, p_2)
        Σ = P * diagm([sd_radial[i], sd_angular[i]]) * P'
        push!(normals, MvNormal(μ, Σ))
    end
    return normals
end