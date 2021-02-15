losses = Float64[10, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 16]

@testset "Never" begin
    @test stopping_time(Never(), losses) == 0
end

@testset "NotANumber" begin
    @test stopping_time(NotANumber(), losses) == 0
    N = 5
    losses2 = fill(123.4, N)
    @test all(reverse(eachindex(losses2))) do j
        losses2[j] = NaN
        stopping_time(NotANumber(), losses2) == j
    end
    losses2 = Float64[1, 2, 3, 1, NaN, 3, 1, 2, 3]
    is_training = Bool[1, 1, 0, 1, 1, 0, 1, 1, 0]
    @test stopping_time(NotANumber(), losses2, is_training) == 2
    losses2 = Float64[1, 2, 3, 1, 2, NaN, 1, 2, 3]
    @test stopping_time(NotANumber(), losses2, is_training) == 2
    losses2 = Float64[1, 2, 3, 1, 2, 3, NaN, 2, 3]
    @test stopping_time(NotANumber(), losses2, is_training) == 3
    losses2 = Float64[1, 2, 3, 1, 2, 3, 1, 2, 3]
    @test stopping_time(NotANumber(), losses2, is_training) == 0
    @test_logs((:info, r"loss updates: 0"),
               (:info, r"state: true"),
               (:info, r"loss updates: 1"),
               (:info, r"state: true"),
               stopping_time(NotANumber(),
                             [NaN, 1],
                             [true, false],
                             verbosity=1))
end

struct SleepyIterator{T}
    iter::T
    t::Float64
end
SleepyIterator(iter; t=0.1) = SleepyIterator(iter, t)

Base.iterate(iter::SleepyIterator) = (sleep(iter.t); iterate(iter.iter))
Base.iterate(iter::SleepyIterator, state) =
    (sleep(iter.t); iterate(iter.iter, state))

@testset "TimeLimit" begin
    @test_throws ArgumentError TimeLimit(t=0)
    @test TimeLimit(1).t == Millisecond(3_600_000)
    @test TimeLimit(t=Day(2)).t == Millisecond(48*3_600_000)
    sleepy_losses = SleepyIterator(losses; t=0.1)
    @test stopping_time(TimeLimit(t=Millisecond(600)), sleepy_losses) == 7
end

@testset "GL" begin
    # constructor:
    @test_throws ArgumentError GL(alpha=0)
    @test GL(alpha=1).alpha === 1.0

    # stopping times:
    n = @test_logs((:info, r"loss updates: 1"),
                   (:info, r"state: \(loss = 10.0, min_loss = 10.0\)"),
                   (:info, r"loss updates: 2"),
                   (:info, r"state: \(loss = 8.0, min_loss = 8.0\)"),
                   (:info, r"loss updates: 3"),
                   (:info, r"state: \(loss = 9.0, min_loss = 8.0\)"),
                   stopping_time(GL(alpha=12), losses, verbosity=1))
    @test n == 3
    @test stopping_time(GL(alpha=20), losses) == 4
    @test stopping_time(GL(alpha=40), losses) == 6
    @test stopping_time(GL(alpha=90), losses) == 11
    @test stopping_time(GL(alpha=110), losses) == 12
    @test stopping_time(GL(alpha=1000), losses) == 0
end

@testset "Patience" begin
    @test_throws ArgumentError Patience(n=0)
    @test stopping_time(Patience(n=6), losses) == 0
    @test stopping_time(Patience(n=5), losses) == 12
    @test stopping_time(Patience(n=4), losses) == 6
    @test stopping_time(Patience(n=3), losses) == 5
    @test stopping_time(Patience(n=2), losses) == 4
    @test stopping_time(Patience(n=1), losses) == 3
end

true
