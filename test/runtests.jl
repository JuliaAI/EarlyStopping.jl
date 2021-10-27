using EarlyStopping, Dates, Test, InteractiveUtils

include("smoke.jl")

@testset "EarlyStopping" begin
    @testset "smoke" begin
        @testset "$C" for C in subtypes(StoppingCriterion)
            @eval @test_criteria $(C())
        end
    end

    @testset "criteria.jl" begin
        include("criteria.jl")
    end

    @testset "disjunction.jl" begin
        include("disjunction.jl")
    end

    @testset "object_oriented_api.jl" begin
        include("object_oriented_api.jl")
    end

    # to complete code coverage:
    @testset "api.jl" begin
        include("api.jl")
    end
end
