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

    done!(stopper, 0.123, training=true)

In these cases, the out-of-sample update must always come after the
corresponding training update. Multiple training updates may preceed
the out-of-sample update.

"""
mutable struct EarlyStopper{S}
    criterion::S
    verbosity::Int
    state
    EarlyStopper(criterion::S; verbosity=0) where S =
        new{S}(criterion, verbosity)
end
EarlyStopper(criteria...; kwargs...) = EarlyStopper(sum(criteria); kwargs...)

for f in [:message, :done]
    eval(quote
         $f(stopper::EarlyStopper) = $f(stopper.criterion, stopper.state)
         end)
end

# defines 2 private functons `done_after_update!(stopper, loss)` and
# `done_after_training_update!(stopper, loss)`:
for f in [:update, :update_training]
    newf = Symbol(string("done_after_", f, "!"))
    eval(quote
         function $newf(stopper::EarlyStopper, loss)
             if isdefined(stopper, :state)
                 stopper.state = $f(stopper.criterion, loss, stopper.state)
             else
                 stopper.state = $f(stopper.criterion, loss)
             end
             return done(stopper)
         end
         end)
end

"""
    done!(stopper::EarlyStopper)
"""
function done!(stopper::EarlyStopper, loss; training=false)
    ret = if training
        done_after_update_training!(stopper, loss)
    else
        done_after_update!(stopper, loss)
    end
    if stopper.verbosity > 0
        suffix = training ? "training " : ""
        loss_str = suffix*"loss"
        @info "$loss_str: $loss\t state: $(stopper.state)"
    end
    return ret
end
