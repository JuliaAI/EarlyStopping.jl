using EarlyStopping, Dates, Test

@testset "criteria.jl" begin
    include("criteria.jl")
end

@testset "disjunction.jl" begin
    include("disjunction.jl")
end

@testset "object_oriented_api.jl" begin
    include("object_oriented_api.jl")
end
