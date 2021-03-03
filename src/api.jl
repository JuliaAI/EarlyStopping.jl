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

needs_loss(::Type) = false
needs_training_losses(::Type) = false

for trait in [:needs_loss, :needs_training_losses]
    eval(:($trait(c::StoppingCriterion) = $trait(typeof(c))))
end
