## ABSTRACT TYPE

abstract type StoppingCriterion end


## FALL BACK METHODS
update(::StoppingCriterion, loss, state=nothing) = state
update_training(::StoppingCriterion, loss, state=nothing) = state

# returns whether it's time to stop:
done(::StoppingCriterion, state) = false

message(criterion::StoppingCriterion, state) = "Stop triggered by "*
    "$criterion stopping criterion. "
