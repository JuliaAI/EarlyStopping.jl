## ABSTRACT TYPE

abstract type StoppingCriterion end


## FALL BACK METHODS

# initialization call is either:
update(::StoppingCriterion, loss) = nothing # state

# ... or:
update_training(::StoppingCriterion, loss) = nothing # state

# subsequent updating:
update(::StoppingCriterion, loss, state) = state
update_training(::StoppingCriterion, loss, state) = state

# returns whether it's time to stop:
done(::StoppingCriterion, state) = false

message(criterion::StoppingCriterion, state) = "Early stop triggered by "*
    "$criterion stopping criterion. "

