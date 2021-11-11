const PRECHELT_REF = "[Prechelt, Lutz (1998): \"Early Stopping"*
    "- But When?\", in *Neural Networks: Tricks of the Trade*, "*
    "ed. G. Orr, Springer.](https://link.springer.com/chapter"*
    "/10.1007%2F3-540-49430-8_3)"

const STOPPING_DOC = "An early stopping criterion for loss-reporting "*
    "iterative algorithms. "

const CUSTOM_ALTERNATIVE_DOC = "For a customizable loss-based stopping "*
    "criterion, use [`WithLossDo`](@ref) or [`WithTrainingLossesDo`](@ref) "*
    "with the `stop_if_true=true` option. "


## NEVER

"""
    Never()

$STOPPING_DOC

Indicates early stopping is to be disabled.

See also [`NotANumber`](@ref), for stopping on encountering `NaN`.

"""
struct Never <: StoppingCriterion end


## OUT OF BOUNDS

"""
    InvalidValue()

$STOPPING_DOC

Stop if a loss (or training loss) is `NaN`, `Inf` or `-Inf` (or, more
precisely, if `isnan(loss)` or `isinf(loss)` is `true`).

$CUSTOM_ALTERNATIVE_DOC

"""
struct InvalidValue <: StoppingCriterion end

_isinf(x) = isinf(x)
_isinf(::Nothing) = false
_isnan(x) = isnan(x)
_isnan(::Nothing) = false

# state = `true` when `NaN`, `Inf` or `-Inf` has been encountered
update(::InvalidValue, loss, state=false) =
    state !== nothing && state || _isinf(loss) || _isnan(loss)
update_training(c::InvalidValue, loss, state) = update(c, loss, state)
done(::InvalidValue, state) = state !== nothing && state

message(::InvalidValue, state) = "Stopping early as `NaN`, "*
    "`Inf` or `-Inf` encountered. "


## TIME LIMIT

"""
    TimeLimit(; t=0.5)

$STOPPING_DOC

Stopping is triggered after `t` hours have elapsed since the stopping
criterion was initiated.

Any Julia built-in `Real` type can be used for `t`. Subtypes of
`Period` may also be used, as in `TimeLimit(t=Minute(30))`.

Internally, `t` is rounded to nearest millisecond.
``
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
update(::TimeLimit, loss, ::Nothing) = now()
update_training(::TimeLimit, loss, ::Nothing) = now()
done(criterion::TimeLimit, state) =
    state === nothing ? false : criterion.t < now() - state


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
with some iterative machine learning algorithm. Then the
*generalization loss* at time `t`, is given by

`` GL_t = 100 (E_t - E_{opt}) \\over |E_{opt}|``

where ``E_{opt}`` is the minimum value of the sequence.

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
update(::GL, loss, ::Nothing) = (loss=loss, min_loss=loss)
update(::GL, loss, state) = (loss=loss, min_loss=min(loss, state.min_loss))
function done(criterion::GL, state)
    if state === nothing
        return false
    else
        gl = generalization_loss(state.loss, state.min_loss)
        return  gl > criterion.alpha
    end
end


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
loss exceeds the threshold ``PQ_T > alpha``, or if the training progress drops
below ``P_j ≤ tol``. Here `k` is the number of training (in-sample) losses used to
estimate the training progress.

## Context and explanation of terminology

The *training progress* at time ``j`` is defined by

`` P_j = 1000 |M - m|/|m| ``

where ``M`` is the mean of the last `k` training losses ``F_1, F_2, …, F_k``
and ``m`` is the minimum value of those losses.

The *progress-modified generalization loss* at time ``t`` is then given by

`` PQ_t = GL_t / P_t``

where ``GL_t`` is the generalization loss at time ``t``; see
[`GL`](@ref).

PQ will stop when the following are true:

1) At least `k` training samples have been collected via
   `done!(c::PQ, loss; training = true)` or `update_training(c::PQ, loss, state)`
2) The last update was an out-of-sample update.
   (`done!(::PQ, loss; training=true)` is always false)
3) The progress-modified generalization loss exceeds the threshold
   ``PQ_t > alpha`` **OR** the training progress stalls ``P_j ≤ tol``.

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

update_training(::PQ, loss, ::Nothing) = PQState([loss, ], true, nothing, nothing)
update(::PQ, loss::T, ::Nothing) where T = PQState(T[], false, loss, loss)

function update_training(criterion::PQ, loss, state)
    training_losses = prepend(state.training_losses, loss, criterion.k)
    return PQState(training_losses, true, state.loss, state.min_loss)
end

function update(::PQ, loss, state)
    min_loss = _min(loss, state.min_loss)
    return PQState(state.training_losses, false, loss, min_loss)
end

function done(criterion::PQ, state)
    state === nothing && return false
    state.loss === nothing && return false
    state.waiting_for_out_of_sample && return false
    length(state.training_losses) < criterion.k && return false
    GL = generalization_loss(state.loss, state.min_loss)
    P = progress(state.training_losses)
    P > criterion.tol || return true
    PQ = GL/P
    return  PQ > criterion.alpha
end


## PATIENCE

# This is UP_s in Prechelt 1998

"""
    Patience(; n=5)

$STOPPING_DOC

A stop is triggered by `n` consecutive increases in the loss.

Denoted "_UP_s" in $PRECHELT_REF.

$CUSTOM_ALTERNATIVE_DOC

"""
struct Patience <: StoppingCriterion
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

update(::Patience, loss, ::Nothing) = (loss=loss, n_increases=0)
@inline function update(::Patience, loss, state)
    old_loss, n = state
    if loss > old_loss
        n += 1
    else
        n = 0
    end
    return (loss=loss, n_increases=n)
end

done(criterion::Patience, state) =
    state === nothing ? false : state.n_increases == criterion.n


## NUMBER SINCE BEST

"""
    NumberSinceBest(; n=6)

$STOPPING_DOC

A stop is triggered when the number of calls to the control, since the
lowest value of the loss so far, is `n`.

$CUSTOM_ALTERNATIVE_DOC

"""
struct NumberSinceBest <: StoppingCriterion
    n::Int
    function NumberSinceBest(n::Int)
        n > 0 ||
            throw(ArgumentError("`n` must be positive. "))
        return new(n)
    end
end
NumberSinceBest(; n=6) = NumberSinceBest(n)

update(::NumberSinceBest, loss, ::Nothing) = (best=loss, number_since_best=0)
@inline function update(::NumberSinceBest, loss, state)
    best, number_since_best = state
    if loss < best
        best = loss
        number_since_best = 0
    else
        number_since_best += 1
    end
    return (best=best, number_since_best=number_since_best)
end

done(criterion::NumberSinceBest, state) =
    state === nothing ? false : state.number_since_best == criterion.n


# # NUMBER LIMIT

"""
    NumberLimit(; n=100)

$STOPPING_DOC

A stop is triggered by `n` consecutive loss updates, excluding
"training" loss updates.

If wrapped in a `stopper::EarlyStopper`, this is the number of calls
to `done!(stopper)`.

"""
struct NumberLimit <: StoppingCriterion
    n::Int
    function NumberLimit(n::Int)
        n > 0 ||
            throw(ArgumentError("`n` must be positive. "))
        return new(n)
    end
end
NumberLimit(; n=100) = NumberLimit(n)

update(criterion::NumberLimit, loss, ::Nothing) = 1
update(::NumberLimit, loss, state) = state+1
done(criterion::NumberLimit, state) =
    state === nothing ? false : state >= criterion.n


# ## THRESHOLD

"""
    Threshold(; value=0.0)

$STOPPING_DOC

A stop is triggered as soon as the loss drops below `value`.

$CUSTOM_ALTERNATIVE_DOC

"""
struct Threshold <: StoppingCriterion
    value::Float64
end
Threshold(; value=0.0) = Threshold(value)

update(::Threshold, loss, state) = loss
done(criterion::Threshold, state) =
    state === nothing ? false : state < criterion.value


"""
    Warmup(c::StoppingCriterion, n)

Wait for `n` updates before checking stopping criterion `c`
"""
struct Warmup{C} <: StoppingCriterion where {C <: StoppingCriterion}
    criterion::C
    n::Int
    function Warmup(criterion::C, n::N) where {C <: StoppingCriterion, N <: Integer}
        n > 0 || throw(ArgumentError("`n` must be positive. "))
        new{C}(criterion, Int(n))
    end
end

# Constructors for Warmup
Warmup() = Warmup(InvalidValue())   # Default for testing
Warmup(c; n = 1) = Warmup(c, n)     # Provide kwargs interface

# Initialize inner state for type-stability, and record first observation
update(c::Warmup, loss, ::Nothing) = update(c, loss)
update(criterion::Warmup, loss) = (1, update(criterion.criterion, loss))

# Catch uninitialized state
update_training(c::Warmup, loss, ::Nothing) = update_training(c, loss)
update_training(c::Warmup, loss) = (1, update_training(c.criterion, loss))

# Handle update vs update_training
update(c::Warmup, loss, state) = _update(update, c, loss, state)
update_training(c::Warmup, loss, state) =
    _update(update_training, c, loss, state)

# Dispatch update and update_training here
function _update(f::Function, criterion::Warmup, loss, state)
    n, inner = state
    n += 1
    if n <= criterion.n
        # Skip inner criterion
        return n, inner
    elseif n == criterion.n+1
        # First step of inner criterion
        return n, f(criterion.criterion, loss)
    else
        # Step inner criterion
        return n, f(criterion.criterion, loss, inner)
    end
end

function done(criterion::Warmup, state)
    # Only check if inner criterion is done after n updates
    state === nothing && return false
    return state[1] <= criterion.n ? false : done(criterion.criterion, state[2])
end

message(c::Warmup, state) = message(c.criterion, state[2])


## NOT A NUMBER (deprecated)

"""
    NotANumber()

$STOPPING_DOC

Stop if a loss of `NaN` is encountered.

**Now deprecated** in favour of [`InvalidValue`](@ref).

"""
struct NotANumber <: StoppingCriterion
    function NotANumber()
        Base.depwarn("`NotANumber()` is deprecated. Use `InvalidValue()` "*
                     "to trap `NaN`, `Inf` or `-Inf`. ", :NotANumber)
        return new()
    end
end


# state = `true` when NaN has been encountered
update(::NotANumber, loss, state) = state !== nothing && state || isnan(loss)
update_training(c::NotANumber, loss, state) = update(c, loss, state)

done(::NotANumber, state) = state !== nothing && state

message(::NotANumber, state) = "Stopping early as NaN encountered. "
