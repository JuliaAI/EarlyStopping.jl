"""
    stopping_time(criterion, losses; verbosity=0)
    stopping_time(criterion, losses, is_training; verbosity=0)

Determine the stopping time for the iterator `losses`, given
`stopping::StoppingCriterion`. Include the `Bool` vector `is_training`
of matching length, when there is a distinction between
"out-of-sample" losses and "training" losses.

If losses completes before a stop, then `0` is returned.

```
julia> stopping_time(NotANumber(), [10.0, 3.0, NaN, 4.0])
3

julia> stopping_time(NotANumber(), [10.0, 3.0, 5.0, 4.0])
0
```
"""
function stopping_time(criterion::EarlyStopper, losses, training; verbosity=0)

    t = 0 # counts regular `update` calls but ignores `update_training` calls
    for (loss, training) in zip(losses, training)
        if !training
            t += 1  # Increment count of out-of-sample updates
        end

        # Update criterion state
        is_done = done!(criterion, loss; training)

        if verbosity > 0
            @info "loss updates: $t"
            @info "state: $(criterion.state)"
        end
        is_done && return t
    end

    # No stopping
    return 0
end

# If training is not provided -> Assume always out-of-sample
stopping_time(criterion, losses; kwargs...) =
    stopping_time(criterion, losses, Iterators.repeated(false); kwargs...)

stopping_time(c::StoppingCriterion, args...; kwargs...) =
    stopping_time(EarlyStopper(c), args...; kwargs...)
