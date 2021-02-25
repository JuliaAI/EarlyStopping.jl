const PRECHELT_REF = "[Prechelt, Lutz (1998): \"Early Stopping"*
    "- But When?\", in *Neural Networks: Tricks of the Trade*, "*
    "ed. G. Orr, Springer.](https://link.springer.com/chapter"*
    "/10.1007%2F3-540-49430-8_3)"

const STOPPING_DOC = "An early stopping criterion for loss-reporting "*
    "iterative algorithms. "


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

"""
struct NotANumber <: StoppingCriterion end

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


## GL

# helper:

generalization_loss(E, E_opt) =  100*(E/abs(E_opt) - one(E_opt))

"""
    GL(; alpha=2.0)

$STOPPING_DOC

A stop is triggered when the (rescaled) generalization loss exceeds
the threshold `alpha`.

**Terminology.** Suppose ``E_1, E_2, ..., E_t`` are a sequence of
losses, for example, out-of-sample estimates of the loss associated
with some iterative machine learing algorithm. Then the
*generalization loss* at time `t`, is given by

`` GL_t = 100 (E_t - E_opt) \\over |E_opt|``

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
done(criterion::GL, state) =
    generalization_loss(state.loss, state.min_loss) > criterion.alpha


## PQ

# helpers:

function prepend(v, x, max_length)
    if length(v) < max_length
        vcat([x, ], v)
    else
        vcat([x, ], v[1:end-1])
    end
end

progress(losses::AbstractVector{T}) where T =
    1000*(mean(losses)/minimum(losses) - one(T))

_min(x, ::Nothing) = x
_min(::Nothing, x) = x
_min(x, y) = min(x, y)

"""
    PQ(; alpha=0.75, k=5, tol=eps(Float64))

A stopping criterion for training iterative supervised learners.

A stop is triggered when Prechelt's progress-modified generalization
loss exceeds the threshold `alpha`, or if the training progress drops
below `tol`. Here `k` is the maximum number of training (in-sample)
losses to be used to estimate the training progress.

**Context and explanation of terminology.** The progress-modified loss
is defined in the following scenario: Estimates, ``E_1, E_2, ...,
E_t``, of the out-of-sample loss of an iterative supervised learner
are being computed, but not necessarily at every iteration. However,
training losses for every iteration *are* being made available
(usually as a by-product of training) which can be used to quantify
recent training progress, as follows.

Fix a time ``j``, corresponding to some out-of-sample loss ``E_j``,
and let ``F_1`` be the corresponding training loss, ``F_2`` the
training loss in the previous interation of the model, ``F_3``, the
training loss two iterations previously, and so on. Let ``K`` denote
the number of model iterations since the last out-of-sample loss
``E_{j-1}`` was computed, or `k`, whichever is the smaller.  Then
the *training progress* at time ``j`` is defined by

`` P_j = 1000 |(M - m) \\over m| ``

where `M` is the mean of the training losses ``F_1, F_2, \\ldots ,
F_K`` and `m` the minimum value of those losses.

The *progress-modified generalization loss* at time ``t`` is given by

`` PQ_t = GL_t \\over P_t``

where ``GL_t`` is the generalization loss at time ``t``; see
[`GL`](@ref).

Reference: $PRECHELT_REF.

"""
struct PQ <: StoppingCriterion
    alpha::Float64
    k::Union{Int,Float64} # could be Inf
    tol::Float64
    function PQ(alpha, k, tol)
        alpha > 0 ||
            throw(ArgumentError("Threshold `alpha` must be positive. "))
        if k < 2 || !(k isa Int || isinf(k))
            throw(ArgumentError("`k` must `Int` or `Inf` and `k > 1`. "))
        end
        tol > 0 ||
            throw(ArgumentError("`tol` must be positive. "))

        return new(alpha, k, tol)
    end
end
PQ(; alpha=0.75, k=5, tol=eps(Float64)) = PQ(alpha, k, tol)

struct PQState{T}
    training_losses::Vector{T}
    waiting_for_out_of_sample::Bool
    loss::Union{Nothing,T}
    min_loss::Union{Nothing,T}
end

function update_training(criterion::PQ, loss)
    training_losses = [loss, ]
    return PQState(training_losses,
                   true,
                   nothing,
                   nothing)
end

update(::PQ, loss) = error("First loss reported to the GL early stopping "*
                           "algorithm must be a training loss. ")

function update_training(criterion::PQ, loss, state)
    training_losses = if state.waiting_for_out_of_sample
        prepend(state.training_losses, loss, criterion.k)
    else
        [loss, ]
    end
    return PQState(training_losses,
                   true,
                   state.loss,
                   state.min_loss)
end

function update(::PQ, loss, state)
    length(state.training_losses) > 1 ||
        error("The PQ stopping criterion requires at least two training "*
              "losses between out-of-sample loss updates. ")
    return PQState(state.training_losses,
                   false,
                   loss,
                   _min(loss, state.min_loss))
end

function done(criterion::PQ, state)
    state.waiting_for_out_of_sample && return false
    GL = generalization_loss(state.loss, state.min_loss)
    P = progress(state.training_losses)
    P > criterion.tol || return true
    PQ = GL/P
    return  PQ > criterion.alpha
end

needs_in_and_out_of_sample(::Type{<:PQ}) = true


## PATIENCE

# This is UP_s in Prechelt 1998

"""
    Patience(; n=5)

$STOPPING_DOC

A stop is triggered by `n` consecutive increases in the loss.

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
Patience(; n=5) = Patience(n)

# Prechelt alias:
const UP = Patience

update(criterion::Patience, loss) = (loss=loss, n_increases=0)
@inline function update(criterion::Patience, loss, state)
    old_loss, n = state
    if loss > old_loss
        n += 1
    else
        n = 0
    end
    return (loss=loss, n_increases=n)
end

done(criterion::Patience, state) = state.n_increases == criterion.n

"""
    MaximumChecks(n::Int)

$STOPPING_DOC

A stop is triggered by `n` consecutive checking of done.

"""
mutable struct MaximumChecks <: StoppingCriterion
    n::Int
    function MaximumChecks(n::Int)
        n > 0 ||
            throw(ArgumentError("The maxchecking level `n` must be positive. "))
        return new(n)
    end
end
MaximumChecks(; n) = MaximumChecks(n)

update(criterion::MaximumChecks, loss) = 1
update_training(criterion::MaximumChecks, loss) = 1
@inline function update(criterion::MaximumChecks, loss, state)
    return state+1
end

done(criterion::MaximumChecks, state) = state == criterion.n
