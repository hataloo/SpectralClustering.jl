using GLMakie
using Random, LinearAlgebra, Statistics, LazySets, Distributions, Clustering, Colors, GeometryBasics
include("SpectralClusteringFunctions.jl")
seed = 149
#seed = Int64(floor(rand()*1e6))
Random.seed!(seed)

#Generate clustered set of points from gaussians
#Centered gaussian
μ = [0,0]
Σ = Matrix{Float64}(undef,2,2)
Σ[1,:], Σ[2,:] = [1,0] , [0,1]
p = MvNormal(μ, Σ)

#Gaussians with μ evenly distributed along a circle of radius R
n = [40, 40, 40, 40, 40, 40] #Number of points to sample per cluster around the circle
normals = getNormals(length(n), 5.0, 1.0, 2.0)
push!(n, 45)
push!(normals, p)
k = [rand(normals[i], n[i])' for i in 1:length(n)]

#Combine into one graph
X = vcat(k...)
V = 1:size(X,1)

K = 4 #Number of clusters in the spectral clustering

#Two points x_1, x_2 have a connection between 
#them if exp(-0.2*norm(x_1 - x_2)) <= max_d
#Repeatedly build the graph and check if the graph is connected.
#If not, decrease max_d. 
L = zeros(2,2)
max_d = 0.5
t = @elapsed begin
    while (eigvals(L)[2] <= 1e-6) #while G is not connected
        global W, max_d, d, D, L
        max_d -= 0.05
        #println("Reducing max_d: ", max_d)
        #W = [1/maximum([norm(X[i,:] - X[j,:]), 0.001]) for i in V, j in V]
        W = [exp(-0.2*norm(X[i,:] - X[j,:])^2) for i in V, j in V]
        for i in V, j in V
            W[i,j] = (W[i,j] .<= max_d) ? 0 : W[i,j]  
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

clusters, clusterIdx = SpectralClustering(K, V, W)


fig = Makie.Figure(resolution = (800,1000))
scatterAx = Makie.Axis(fig[1,1])
alphabet = repeat('A':'Z', Int64(ceil((sum(n)+1) / 26)))
markers = alphabet
colors = distinguishable_colors(length(k) + K, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
scattercolors = colors[1:length(k)]
clustercolors = colors[(length(k)+1):(length(k)+K)]
iter = 0
for (i,cluster) in enumerate(k)
    Makie.scatter!(scatterAx, cluster, marker = markers[i], markersize = 15, color = scattercolors[i])
    #for j in 1:size(cluster,1)
    #    global iter 
    #    iter = iter + 1
    #    Makie.scatter!(scatterAx, cluster[j,:]', marker = markers[i], markersize = 15, color = clustercolors[clusterIdx[iter]])
    #end
end

#Draw edges
connection_vector = [[X[i,:], X[j,:]] for i in V for j in 1:i if (W[i,j] > 1e-9)]
connection_strength = [W[i,j] for i in V for j in 1:i if (W[i,j] > 1e-9)]
connection_list = zeros(2*size(connection_vector,1),2)
for i in 1:size(connection_vector,1)
    connection_list[2*i-1,:] = connection_vector[i][1]
    connection_list[2*i,:] = connection_vector[i][2]
end
Makie.linesegments!(connection_list[:,1], connection_list[:,2], 
        linewidth = 0.3*connection_strength, alpha = 0.1)


markers = 'A':'Z'
for (i, C) in enumerate(clusters)
    hull = convex_hull([X[i,:] for i in C])
    hull_points = [Point2(s) for s in hull]
    Makie.lines!(push!(hull_points, hull_points[1]), color = clustercolors[i], linewidth = 2.0)
    scatter_points = [Point2(X[i,:]) for i in C]
    #Makie.scatter!(scatterAx, scatter_points, marker = markers[i], markersize = 20, color = colors[i])
end


#contFig = Makie.Figure()
#contourAx = Makie.Axis(contFig[1,1])
contourAx = Makie.Axis(fig[2,1])
for (i,cluster) in enumerate(k)
    Makie.scatter!(contourAx, normals[i].μ', marker = markers[i], markersize = 15, color = scattercolors[i])
end
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