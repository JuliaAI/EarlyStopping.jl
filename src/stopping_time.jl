"""
    stopping_time(criterion, losses; verbosity=0)
    stopping_time(criterion, losses, is_training; verbosity=0)

Determine the stopping time for the iterator `losses`, for the specified stopping
`criterion`. Include the `Bool` vector `is_training` of matching length, when there is a
distinction between "out-of-sample" losses and "training" losses.

If losses completes before a stop, then `0` is returned.

```
julia> stopping_time(NotANumber(), [10.0, 3.0, NaN, 4.0])
3

julia> stopping_time(NotANumber(), [10.0, 3.0, 5.0, 4.0])
0
```
"""
function stopping_time(stopper::EarlyStopper, losses, training; verbosity=0)

    t = 0 # counts regular `update` calls but ignores `update_training` calls
    for (loss, training) in zip(losses, training)
        if !training
            t += 1  # Increment count of out-of-sample updates
        end

        # Update stopper state
        is_done = done!(stopper, loss; training = training)

        if verbosity > 0
            @info "loss updates: $t"
            @info "state: $(stopper.state)"
        end
        is_done && return t
    end

    # No stopping
    return 0
end

# If training is not provided -> Assume always out-of-sample
stopping_time(stopper::EarlyStopper, losses; kwargs...) =
    stopping_time(stopper, losses, Iterators.repeated(false); kwargs...)

stopping_time(c, args...; kwargs...) =
    stopping_time(EarlyStopper(c, verbosity=-1), args...; kwargs...)
