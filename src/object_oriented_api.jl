"""

    EarlyStopper(c...; verbosity=0)

Instantiate an object for tracking whether one or more stopping
criterion `c` apply, given a sequence of losses.

For a list of possible criterion, do
`subtypes(EarlyStopping.StoppingCriterion)`.

### Sample usage

    stopper = EarlyStopper(Patience(1), NotANumber())
    done!(stopper, 0.123) # false
    done!(stopper, 0.234) # true

    julia> message(stopper)
    "Early stop triggered by Patience(1) stopping criterion. "

### Training losses

For criteria tracking both an "out-of-sample" loss and a "training"
loss (eg, stopping criterion of type `PQ`), specify `training=true` if
the update is for training, as in

    done!(stopper, 0.123; training=true)

In these cases, the out-of-sample update must always come after the
corresponding training update. Multiple training updates may precede
the out-of-sample update.

"""
mutable struct EarlyStopper{S}
    criterion::S
    verbosity::Int
    state
    EarlyStopper(criterion::S; verbosity=0) where S =
        new{S}(criterion, verbosity, nothing)
end
EarlyStopper(criteria...; kwargs...) = EarlyStopper(sum(criteria); kwargs...)

# Dispatch message, done, update and update_training to wrapped criterion
for f in [:message, :done]
    @eval $f(stopper::EarlyStopper) = $f(stopper.criterion, stopper.state)
end
for f in [:update, :update_training]
    @eval $f(stopper::EarlyStopper, loss) =
        $f(stopper.criterion, loss, stopper.state)
end

"""
    done!(stopper::EarlyStopper, loss; training = false)
"""
function done!(stopper::EarlyStopper, loss; training=false)
    if training
        stopper.state = update_training(stopper, loss)
    else
        stopper.state = update(stopper, loss)
    end
    if stopper.verbosity > 0
        suffix = training ? "training " : ""
        loss_str = suffix*"loss"
        @info "$loss_str: $loss\t state: $(stopper.state)"
    end
    return done(stopper)
end

"""
    reset!(stopper::EarlyStopper)
    reset!(stopper::EarlyStopper, state)

Reset a stopper to it's uninitialized state or to a particular state
"""
reset!(stopper::EarlyStopper) = stopper.state = nothing
reset!(stopper::EarlyStopper, state) = stopper.state = state
