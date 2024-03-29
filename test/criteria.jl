losses = Float64[10, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 16]

# codecov:
@test EarlyStopping._min(nothing, 5) == 5

@testset "Never" begin
    @test stopping_time(Never(), losses) == 0
end

@testset "InvalidValue" begin
    @test stopping_time(InvalidValue(), losses) == 0
    N = 5
    losses2 = fill(123.4, N)
    @test all(reverse(eachindex(losses2))) do j
        losses2[j] = NaN
        stopping_time(InvalidValue(), losses2) == j
    end

    is_training = map(x -> x%3 > 0, 1:length(losses))
    @test stopping_time(InvalidValue(), losses, is_training) == 0
    for n = 1:2:length(losses)
        n_stop = sum(!, is_training[1:n])
        losses2 = copy(losses)
        losses2[n] = Inf
        @test stopping_time(InvalidValue(), losses2, is_training) == n_stop
        losses2[n] = NaN
        @test stopping_time(InvalidValue(), losses2, is_training) == n_stop
    end
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

    state = EarlyStopping.update_training(c, 10.0)
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

@testset "NumberSinceBest" begin
    @test_throws ArgumentError NumberSinceBest(n=0)
    @test stopping_time(NumberSinceBest(n=6), losses) == 8
    @test stopping_time(NumberSinceBest(n=5), losses) == 7
    @test stopping_time(NumberSinceBest(n=4), losses) == 6
    @test stopping_time(NumberSinceBest(n=3), losses) == 5
    @test stopping_time(NumberSinceBest(n=2), losses) == 4
    @test stopping_time(NumberSinceBest(n=1), losses) == 3

    losses2 = Float64[10, 9, 8, 9, 10, 7, 10, 10, 10, 10]
    @test stopping_time(NumberSinceBest(n=2), losses2) == 5
    @test stopping_time(NumberSinceBest(n=3), losses2) == 9
end

@testset "NumberLimit" begin
    @test_throws ArgumentError NumberLimit(n=0)

    for i in 1:length(losses)
        @test stopping_time(NumberLimit(i), losses) == i
    end
end

@testset "Threshold" begin
    @test Threshold().value == 0.0
    stopping_time(Threshold(2.5), Float64[12, 32, 3, 2, 5, 7]) == 4
end

@testset "robustness to first loss being a training loss" begin
    criteria = filter(subtypes(StoppingCriterion)) do C
        C != NotANumber # as deprecated
    end
    for C in criteria
        losses = float.(4:-1:1)
        is_training = [true, true, false, false]
        stopping_time(C(), losses, is_training)
    end
end

@testset "Warmup" begin
    @test_throws ArgumentError Warmup(Patience(), 0)
    for n in 1:(length(losses)-1)
        @test stopping_time(Warmup(NumberLimit(1), n), losses) == n+1
    end

    # Test message
    @testset "message" begin
        stopper = Warmup(Patience(2); n = 2)
        stopper_ref = Warmup(Patience(2), 2)
        state, state_ref = nothing, nothing
        for loss = losses
            state = EarlyStopping.update(stopper, loss, state)
            state_ref = EarlyStopping.update(stopper_ref, loss, state_ref)
            @test message(stopper, state) == message(stopper_ref, state_ref)
        end
    end

    @testset "training" begin
        stopper = Warmup(PQ(), 3)
        is_training = map(x -> x%3 > 0, 1:length(losses))

        # Feed 2 training losses + 1 non-training to criteria with/without
        stop_time = stopping_time(stopper, losses, is_training)
        ref_stop_time = stopping_time(stopper.criterion, losses[3:end], is_training)

        # PQ only counts training loss updates
        @test round(stop_time/3, RoundUp) == ref_stop_time
    end

    @testset "integration" begin
        @test_criteria Warmup(Patience())
        @test_criteria Warmup(NumberSinceBest())
        @test_criteria Warmup(Patience(3) + InvalidValue())
    end
end


# # DEPRECATED

@testset "NotANumber" begin
    @test_deprecated criterion = NotANumber()
    @test stopping_time(criterion, losses) == 0
    N = 5
    losses2 = fill(123.4, N)
    @test all(reverse(eachindex(losses2))) do j
        losses2[j] = NaN
        stopping_time(criterion, losses2) == j
    end
    is_training = map(x -> x%3 > 0, 1:length(losses))
    @test stopping_time(criterion, losses, is_training) == 0
    for n = 1:2:length(losses)
        losses2 = copy(losses); losses2[n] = NaN
        @test stopping_time(criterion, losses2, is_training) == sum(!, is_training[1:n])
    end
end


true
