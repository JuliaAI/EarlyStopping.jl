c1 = Patience(1)
c2 = NotANumber()
c3 = TimeLimit(t=100)

@test Disjunction(c1) == c1

d = c1 + c2 + Never() + c3 + c1
show(d)

@test sum(Disjunction[]) == Never()
@test Never() in c1

@testset "_criteria" begin
    criteria = EarlyStopping._criteria(d)
    @test length(criteria) == 3
    @test issubset([c1, c2, c3], criteria)
end

@testset "stoppping times" begin
    d2 = Patience(3) + NotANumber()
    @test stopping_time(d2, [12.0, 10.0, 11.0, 12.0, 13.0, NaN]) == 5
    @test stopping_time(d2, [NaN, 12.0, 10.0, 11.0, 12.0, 13.0]) == 1
end

@testset "message" begin
    state = EarlyStopping.update(d, NaN)
    @test EarlyStopping.message(d, state) ==
        "Stopping early as NaN encountered. "
end

state = EarlyStopping.update(d, 1.0)
state = EarlyStopping.update(d, 2.0, state)
@test EarlyStopping.message(d, state) ==
    "Early stop triggered by Patience(1) stopping criterion. "
