## ABSTRACT TYPE

abstract type StoppingCriterion end


## FALL BACK METHODS
update(::StoppingCriterion, loss, state=nothing) = state
update_training(::StoppingCriterion, loss, state=nothing) = state

# returns whether it's time to stop:
done(criterion, state) = false

"""
    public_state(criterion, state)

Transform the `state` associated with some stopping `criterion` (or other control from
IterationControl.jl) into a user-inspectable form. This form is stable, while `state` is
private API.

# New control implementations

This method must return a named tuple. A fallback returns `state`.
"""
public_state(criterion, state) = state

message(criterion, state) = "Stop triggered by "*
    "$criterion stopping criterion. "
