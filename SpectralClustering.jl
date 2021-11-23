using GLMakie
using Random, LinearAlgebra, Statistics, LazySets, Distributions, Clustering, Colors, GeometryBasics
include("SpectralClusteringFunctions.jl")
seed = 122
#seed = Int64(floor(rand()*1e6))
Random.seed!(seed)

#Generate clustered set of points from gaussians
#Centered gaussian
μ = [0,0]
Σ = Matrix{Float64}(undef,2,2)
Σ[1,:], Σ[2,:] = [1,0] , [0,1]
p = MvNormal(μ, Σ)

#Gaussians with μ evenly distributed along a circle of radius R
n = [40, 40, 40, 20, 40, 0] #Number of points to sample per cluster around the circle
normals = getNormals(length(n), 5.0, 1.0, 2.0)
push!(n, 45)
push!(normals, p)
k = [rand(normals[i], n[i])' for i in 1:length(n)]

#Combine into one graph
X = vcat(k...)
V = 1:size(X,1)

K = 6 #Number of clusters in the spectral clustering

#Two points x_1, x_2 have a connection between 
#them if norm(x_1 - x_2) <= max_d
#Repeatedly build the graph and check if the graph is connected.
#If not, increase max_d. 
L = zeros(2,2)
max_d = 0.0
t = @elapsed begin
    while (eigvals(L)[2] <= 1e-6) #while G is not connected
        global W, max_d, d, D, L
        max_d += 0.2
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

#Can verify that using minNcut version is equivalent to SpectralClustering.
_, _, _, ϕ = getSpectralUtility(V, W)
τ, S, Sᶜ = minNcut(ϕ[:,2], W, V)

clusters = SpectralClustering(K, V, W)


fig = Makie.Figure(resolution = (800,1000))
scatterAx = Makie.Axis(fig[1,1])
alphabet = repeat('A':'Z', Int64(ceil((sum(n)+1) / 26)))
markers = alphabet
colors = distinguishable_colors(length(k) + K, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
scattercolors = colors[1:length(k)]
clustercolors = colors[(length(k)+1):(length(k)+K)]
for (i,cluster) in enumerate(k)
    Makie.scatter!(scatterAx, cluster, marker = markers[i], markersize = 15, color = scattercolors[i])
end

#Draw edges
connection_vector = [[X[i,:], X[j,:]] for i in V for j in 1:i if (W[i,j] > 1e-6)]
connection_list = zeros(2*size(connection_vector,1),2)
for i in 1:size(connection_vector,1)
    connection_list[2*i-1,:] = connection_vector[i][1]
    connection_list[2*i,:] = connection_vector[i][2]
end
Makie.linesegments!(connection_list[:,1], connection_list[:,2], 
        linewidth = 0.3, alpha = 0.1)


markers = 'A':'Z'
for (i, C) in enumerate(clusters)
    hull = convex_hull([X[i,:] for i in C])
    hull_points = [Point2(s) for s in hull]
    Makie.lines!(push!(hull_points, hull_points[1]), color = clustercolors[i])
    scatter_points = [Point2(X[i,:]) for i in C]
    #Makie.scatter!(scatterAx, scatter_points, marker = markers[i], markersize = 20, color = colors[i])
end


contourAx = Makie.Axis(fig[2,1])
#for (μ, Σ, p) in [[μ_1, Σ_1, p_1], [μ_2, Σ_2, p_2]]
#    xs = LinRange(μ[1]-Σ[1,1]*2, μ[1]+Σ[1,1]*2, 100)
#    ys = LinRange(μ[2]-Σ[2,2]*2, μ[2]+Σ[2,2]*2, 100)
#    zs = [pdf(p, [x,y]) for x in xs, y in ys]
#    contour!(xs,ys,zs)
#end

for (i,p) in enumerate(normals)
    if n[i] == 0 continue end
    local μ, Σ
    μ, Σ = p.μ, p.Σ
    xs = LinRange(μ[1]-Σ[1,1]*2, μ[1]+Σ[1,1]*2, 100)
    ys = LinRange(μ[2]-Σ[2,2]*2, μ[2]+Σ[2,2]*2, 100)
    zs = [pdf(p, [x,y]) for x in xs, y in ys]
    contour!(xs,ys,zs)

end

fig