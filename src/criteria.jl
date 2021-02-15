# Includes stopping criterion surveyed in Prechelt, Lutz (1998):
# "Early Stopping - But When?", in "Neural Networks: Tricks of the
# Trade", ed. G. Orr, Springer.

# https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3

# criterion                     | notation in Prechelt
# ------------------------------|--------------------------------
# `Never()`                     | -
# `NotANumber()`                | -
# `TimeLimit(t=...)`            | -
# `GL(alpha...)`                | ``GL_α``
# `PQ(alpha...)`                | ``PQ_α``
# `Patience(n=...)`             | ``UP_s``



# "loss" can have any type for which `<` is defined; it is allowed to
# be negative (eg, Brier loss is sometimes defined that way).

const PRECHELT_REF = "[Prechelt, Lutz (1998): \"Early Stopping"*
    "- But When?\", in *Neural Networks: Tricks of the Trade*, "*
    "ed. G. Orr, Springer.](https://link.springer.com/chapter"*
    "/10.1007%2F3-540-49430-8_3)"

const STOPPING_DOC = "A stopping criterion for training
    "* "iterative statistical models. "


## NEVER

"""
    Never()

$STOPPING_DOC

Indicates early stopping is to be disabled.

See also [`NotANumber`](@ref), for stopping on encountering `NaN`.

"""
struct Never <: StoppingCriterion end


## NOT A NUMBER

"""
    NotANumber()

$STOPPING_DOC

Stop if a loss of `NaN` is encountered.
`training_only=false`.

"""
struct NotANumber end

# state = `true` when NaN has been encountered

update(::NotANumber, loss) = isnan(loss)
update_training(::NotANumber, loss) = isnan(loss)

update(::NotANumber, loss, state) = state || isnan(loss)
update_training(::NotANumber, loss, state) = state || isnan(loss)

done(::NotANumber, state) = state

message(::NotANumber, state) = "Stopping early as NaN encountered. "


## TIME LIMIT

"""
    TimeLimit(; t=0.5)

$STOPPING_DOC

Stopping is triggered after `t` hours have elapsed since the stopping
criterion was initiated.

Any Julia built-in `Real` type can be used for `t`. Subtypes of
`Period` may also be used, as in `TimeLimit(t=Minute(30))`.

Internally, `t` is rounded to nearest millisecond.

"""
struct TimeLimit <: StoppingCriterion
    t::Millisecond
    function TimeLimit(t::Millisecond)
        t > Millisecond(0) ||
            throw(ArgumentError("Time limit `t` must be positive. "))
        return new(t)
    end
end
TimeLimit(t::T) where T <: Period = TimeLimit(convert(Millisecond, t))
# for t::T a "numeric" time in hours; assumes `round(Int, ::T)` implemented:
TimeLimit(t) = TimeLimit(round(Int, 3_600_000*t) |> Millisecond)
TimeLimit(; t =Minute(30)) = TimeLimit(t)

# state = time at initialization

update(::TimeLimit, loss) = now()
update_training(::TimeLimit, loss) = now()
update(::TimeLimit, loss, state) = state
done(criterion::TimeLimit, state) = begin
    criterion.t < now() - state
end

## GENERALIZATION LOSS

# This is GL_α in Prechelt 1998

"""
    GL(; alpha=2.0)

$STOPPING_DOC

A stop is triggered when Prechelt's generalization loss exceeds the
threshold `alpha`. Prechelt's generalization loss, for a sequence
`E_1, E_2, ..., E_t` of out-of-sample estimates of the loss, is

`` GL = 100*(E_t - E_opt)/|E_opt|``

where `E_opt` is the minimum value of the sequence.

Reference: $PRECHELT_REF.

"""
struct GL <: StoppingCriterion
    alpha::Float64
    function GL(alpha)
        alpha > 0 ||
            throw(ArgumentError("Threshold `alpha` must be positive. "))
        return new(alpha)
    end
end
GL(; alpha=2.0) = GL(alpha)

update(::GL, loss) = (loss=loss, min_loss=loss)
update(::GL, loss, state) = (loss=loss, min_loss=min(loss, state.min_loss))
@inline function done(criterion::GL, state)
    E, E_opt = state
    GL = 100*(E/abs(E_opt) - one(E_opt))
    return GL > criterion.alpha
end


## PATIENCE

# This is UP_s in Prechelt 1998

"""
    Patience(; n=5)

$STOPPING_DOC

A stop is triggered by `n` consecutive deteriorations in the out-of-sample
performance.

Denoted "_UP_s" in $PRECHELT_REF.

"""
mutable struct Patience <: StoppingCriterion
    n::Int
    function Patience(n::Int)
        n > 0 ||
            throw(ArgumentError("The patience level `n` must be positive. "))
        return new(n)
    end
end
Patience(; n=1) = Patience(n)

# Prechelt alias:
const UP = Patience

update(criterion::Patience, loss) = (loss=loss, num_drops=0)
@inline function update(criterion::Patience, loss, state)
    old_loss, n = state
    if loss > old_loss
        n += 1
    else
        n = 0
    end
    return (loss=loss, num_drops=n)
end

done(criterion::Patience, state) = state.num_drops' == criterion.n
