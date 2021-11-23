using GLMakie, Random, LinearAlgebra, Statistics, LazySets, Distributions, Clustering, Colors
seed = 127
#seed = Int64(floor(rand()*1e6))
Random.seed!(seed)

#Generate two clusters 
n_1,n_2 = 100, 50

μ_1 = [0,0]
Σ_1 = Matrix{Float64}(undef,2,2)
Σ_1[1,:], Σ_1[2,:] = [1,-0.9] , [-0.9,1]
p_1 = MvNormal(μ_1, Σ_1)
k_1 = rand(p_1, n_1)'

μ_2 = [2,2]
Σ_2 = Matrix{Float64}(undef,2,2)
Σ_2[1,:], Σ_2[2,:] = [1,0] , [0,1]
p_2 = MvNormal(μ_2, Σ_2)
k_2 = rand(p_2, n_2)'

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
normals = getNormals(3, 1.0, 2.0, 1.0)
n = [30, 30]
k = []
#Combine into one graph
X = vcat(k_1,k_2)
V = 1:size(X,1)

#Two points x_1, x_2 have a connection between 
#them if norm(x_1 - x_2) <= max_d
#Repeatedly build the graph and check if the graph is connected.
#If not, increase max_d. 
L = zeros(2,2)
max_d = 0.0
t = @elapsed begin
    while (eigvals(L)[2] <= 1e-6) #while G is not connected
        global W, max_d, d, D, L
        max_d += 0.1
        W = [norm(X[i,:] - X[j,:]) for i in V, j in V]
        for i in V, j in V
            W[i,j] = (W[i,j] .>= max_d) ? 0 : W[i,j]  
        end
        d = [sum(W[i,:]) for i in V]
        D = diagm(d)
        L = D - W
    end
end
println("Time to build graph: ", t, " sec")


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

_, _, _, ϕ = getSpectralUtility(V, W)

τ, S, Sᶜ = minNcut(ϕ[:,2], W, V)

S, Sᶜ = SpectralClustering(2, V, W)
K = 3
clusters = SpectralClustering(K, V, W)
#τ = mean(ϕ₂)
#S = V[ϕ₂ .<= τ]
#Sᶜ = V[ϕ₂ .> τ]


fig = GLMakie.Figure(resolution = (800,1000))
scatterAx = Axis(fig[1,1])
alphabet = repeat('A':'Z', Int64(ceil((n_1+n_2+1) / 26)))
m_1 = (alphabet)[1:(n_1)]
m_2 = (alphabet)[(n_1+1):(n_1+n_2)]
GLMakie.scatter!(scatterAx, k_1, marker = :xcross, markersize = 15, color = :red)
GLMakie.scatter!(scatterAx, k_2, marker = :star5, markersize = 15, color = :purple)

#Draw edges
connection_vector = [[X[i,:], X[j,:]] for i in V for j in 1:i if (W[i,j] > 1e-6)]
connection_list = zeros(2*size(connection_vector,1),2)
for i in 1:size(connection_vector,1)
    connection_list[2*i-1,:] = connection_vector[i][1]
    connection_list[2*i,:] = connection_vector[i][2]
end
GLMakie.linesegments!(connection_list[:,1], connection_list[:,2], 
        linewidth = 0.3, alpha = 0.1)


#S_hull = convex_hull([X[i,:] for i in S])
#S_points = [Point2(s) for s in S_hull]
#Sᶜ_hull = convex_hull([X[i,:] for i in Sᶜ])
#Sᶜ_points = [Point2(s) for s in Sᶜ_hull]
#GLMakie.poly!(Polygon(Sᶜ_points), alpha = 0.1)
#GLMakie.poly!(Polygon(S_points), transparency = true)
#GLMakie.lines!(push!(S_points,S_points[1]), color = :red)
#GLMakie.lines!(push!(Sᶜ_points,Sᶜ_points[1]), color = :purple)


colors = distinguishable_colors(K, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
markers = 'A':'Z'
for (i, C) in enumerate(clusters)
    hull = convex_hull([X[i,:] for i in C])
    hull_points = [Point2(s) for s in hull]
    GLMakie.lines!(push!(hull_points, hull_points[1]), color = colors[i])
    scatter_points = [Point2(X[i,:]) for i in C]
    #GLMakie.scatter!(scatterAx, scatter_points, marker = markers[i], markersize = 20, color = colors[i])
end


contourAx = Axis(fig[2,1])
for (μ, Σ, p) in [[μ_1, Σ_1, p_1], [μ_2, Σ_2, p_2]]
    xs = LinRange(μ[1]-Σ[1,1]*2, μ[1]+Σ[1,1]*2, 100)
    ys = LinRange(μ[2]-Σ[2,2]*2, μ[2]+Σ[2,2]*2, 100)
    zs = [pdf(p, [x,y]) for x in xs, y in ys]
    contour!(xs,ys,zs)
end


fig