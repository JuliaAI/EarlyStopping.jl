losses = Float64[10, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 16]

# codecov:
@test EarlyStopping._min(nothing, 5) == 5

@testset "Never" begin
    @test stopping_time(Never(), losses) == 0
    @test !EarlyStopping.needs_loss(Never())
    @test !EarlyStopping.needs_training_losses(Never())
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
    @test EarlyStopping.needs_loss(NotANumber())
    @test !EarlyStopping.needs_training_losses(NotANumber())
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
    # codecov:
    @test EarlyStopping.update_training(TimeLimit(), 42.0) <= now()
    @test !EarlyStopping.needs_loss(TimeLimit())
    @test !EarlyStopping.needs_training_losses(TimeLimit())
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

    @test EarlyStopping.needs_loss(GL())
    @test !EarlyStopping.needs_training_losses(GL())
end

@testset "PQ" begin
    v = [:c, :b, :a]
    @test EarlyStopping.prepend(v, :d, 4) == [:d, :c, :b, :a]
    @test EarlyStopping.prepend(v, :d, 3) == [:d, :c, :b]
    @test EarlyStopping.progress(Float64[2, 3, 6, 7]) ≈ 1000*(4.5 - 2)/2
    @test_throws ArgumentError PQ(alpha=0)
    @test_throws ArgumentError PQ(k=-Inf)
    @test_throws ArgumentError PQ(k=1.2)
    @test_throws ArgumentError PQ(k=1)

    c = PQ(alpha=10, k=2)

    # first update must be training:
    @test_throws Exception EarlyStopping.update(c, 1.0)

    state = EarlyStopping.update_training(c, 10.0)
    # at least two training updates before out-of-sample update:
    @test_throws Exception EarlyStopping.update(c, state, 10.0)

    state = EarlyStopping.update_training(c, 10.0, state)
    state = EarlyStopping.update(c, 10.0, state)
    @test EarlyStopping.done(c, state) # progress = 0

    state = EarlyStopping.update_training(c, 10.0, state)
    # can't be done if last update was a training update:
    @test !EarlyStopping.done(c, state)

    #                 k=2                progress GL    PQ    t
    losses2 = [9.5, 9.3, 10,            # 10.8     0     0     1
              9.3, 9.1, 8.9, 8,        # 11.2     0     0     2
              8.3, 8.4, 9,             # 6.02     12.5  2.08  3
              9.9, 9.5, 10,            # 21.2     25.0  1.18  4
              10.6, 10.4, 11,          # 9.61     37.5  3.90  5
              8888, 11.8, 11.7, 12,    # 4.27     50.0  11.7  6
              11.6, 11.4, 12,          # 8.77     50.0  5.70  7
              12.2, 12.1, 13,          # 4.1      62.5  15.2  8
              14.5, 14.1, 14,          # 14.2     75    5.28  9
              13.9, 13.7, 15,          # 7.30     87.5  11.9  10
              12.5, 12.3, 16,          # 8.13     100   12.3  11
              11.2, 11.0, 17,          # 9.09     112.5 12.4  12
              10.5, 10.3, 16]          # 9.71     100   10.3  13
    is_training = Bool[1, 1, 0,
                       1, 1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0,
                       1, 1, 0]

    @test stopping_time(PQ(alpha=2.0, k=2), losses2, is_training) == 3
    @test stopping_time(PQ(alpha=3.8, k=2), losses2, is_training) == 5
    @test stopping_time(PQ(alpha=11.6, k=2), losses2, is_training) == 6
    @test stopping_time(PQ(alpha=15.1, k=2), losses2, is_training) == 8
    @test stopping_time(PQ(alpha=15.3, k=2), losses2, is_training) == 0

    @test EarlyStopping.needs_loss(PQ())
    @test EarlyStopping.needs_training_losses(PQ())
end

@testset "Patience" begin
    @test_throws ArgumentError Patience(n=0)
    @test stopping_time(Patience(n=6), losses) == 0
    @test stopping_time(Patience(n=5), losses) == 12
    @test stopping_time(Patience(n=4), losses) == 6
    @test stopping_time(Patience(n=3), losses) == 5
    @test stopping_time(Patience(n=2), losses) == 4
    @test stopping_time(Patience(n=1), losses) == 3

    @test EarlyStopping.needs_loss(Patience())
    @test !EarlyStopping.needs_training_losses(Patience())
end

@testset "CountLimit" begin
    @test_throws ArgumentError CountLimit(n=0)

    for i in 1:length(losses)
        @test stopping_time(CountLimit(i), losses) == i
    end

    @test !EarlyStopping.needs_loss(CountLimit())
    @test !EarlyStopping.needs_training_losses(CountLimit())

end

@testset "Threshold" begin
    @test Threshold().value == 0.0
    stopping_time(Threshold(2.5), Float64[12, 32, 3, 2, 5, 7]) == 4
    @test EarlyStopping.needs_loss(Threshold())
    @test !EarlyStopping.needs_training_losses(Threshold())
end

@testset "robustness to first loss being a training loss" begin
    for C in subtypes(StoppingCriterion)
        losses = float.(4:-1:1)
        is_training = [true, true, false, false]
        stopping_time(C(), losses, is_training)
    end
end

true
