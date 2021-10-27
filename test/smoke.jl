"""
    test_criteria StoppingCriterion()

Runs a series of tests to check functionality of a StoppingCriterion

- `update(::StoppingCriterion, loss, state=nothing)` is defined
- Checks that `done(::StoppingCriterion, nothing)` is `false`
- `done(::StoppingCriterion, state)` can handle state after `update` or `update_training`
- `message(::StoppingCriterion, state)` can handle state after `update` or `update_training`

"""
macro test_criteria(criteria)
    quote
        @test $criteria isa StoppingCriterion
        @testset "state is Nothing" begin
            loss = rand()
            using EarlyStopping: update, update_training

            # Check update for nothing state
            c1 = update($criteria, loss)
            c2 = update($criteria, loss, nothing)
            @test compare_state(c1, c2)

            # Check update_training for nothing state
            c1 = update_training($criteria, loss)
            c2 = update_training($criteria, loss, nothing)
            @test compare_state(c1, c2)
        end

        # Check that done can be called after `update` or `update_training`
        @testset "done" begin
            loss = rand()
            using EarlyStopping: done, update, update_training
            @test done($criteria, nothing) == false
            @test done($criteria, update($criteria, loss)) isa Bool
            @test done($criteria, update_training($criteria, loss)) isa Bool

            # Training then out-of-sample
            loss2 = rand()
            s = update_training($criteria, loss2, update($criteria, loss))
            @test done($criteria, s) isa Bool

            # Out of sample then
            @test done($criteria, update($criteria, loss2, update_training($criteria, loss))) isa Bool
        end

        # Check that `message` can handle state after `update` or `update_training`
        @testset "message" begin
            loss = rand()
            using EarlyStopping: update, update_training
            @test message($criteria, update($criteria, loss)) isa String
            @test message($criteria, update_training($criteria, loss)) isa String
        end
    end
end

"""
    compare_state(s1, s2)

Helper method to check if StoppingCriterion states `s1` and `s2` are effectively
equivalent.
"""
compare_state(s1, s2) = s1 == s2
compare_state(s1::DateTime, s2::DateTime) = abs(s1 - s2) < Millisecond(10)

# Extend == for PQState
import Base: ==
function ==(s1::EarlyStopping.PQState, s2::EarlyStopping.PQState)
    length(s1.training_losses) == length(s2.training_losses) &&
    all(s1.training_losses .== s2.training_losses) &&
    s1.waiting_for_out_of_sample == s2.waiting_for_out_of_sample &&
    s1.loss == s2.loss &&
    s1.min_loss == s2.min_loss
end
