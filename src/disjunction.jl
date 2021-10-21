"""
    Disjunction(criteria...)

$STOPPING_DOC

Combines the specified stopping `criteria` dijunctively: if any one of
the criteria applies, then stop.

**Syntactic sugar.** `c1 + c2 + ...` is equivalent to
  `Disjunction(c1, c2, ...)`.

"""
struct Disjunction{A,B} <: StoppingCriterion
    a::A
    b::B
    function Disjunction(a::A, b::B) where {A, B}
        a in b && return b # `in` defined post-facto below
        b in a && return a
        return new{A,B}(a, b)
    end
end

Disjunction() = Never()
Disjunction(a) = a
Disjunction(a, b, c...) = Disjunction(Disjunction(a,b), c...)

for f in [:update, :update_training]
    eval(quote
         $f(d::Disjunction, loss, ::Nothing) =
             (a = $f(d.a, loss), b = $f(d.b, loss))
         $f(d::Disjunction, loss, state) =
             (a = $f(d.a, loss, state.a),
              b = $f(d.b, loss, state.b))
         end)
end


## RECURSION TO EXTRACT COMPONENTS AND TEST MEMBERSHIP

# fallback for atomic criteria:
_push!(criteria, criterion) = push!(criteria, criterion)

# disjunction:
function _push!(criteria, d::Disjunction)
    _push!(criteria, d.a)
    _push!(criteria, d.b)
end

_criteria(c::StoppingCriterion) = _push!(StoppingCriterion[], c)

Base.in(::Never, ::StoppingCriterion) = true
Base.in(::Never, ::Disjunction) = true
Base.in(c1::StoppingCriterion, c2::StoppingCriterion) = c1 == c2
Base.in(c::StoppingCriterion, d::Disjunction) = c in _criteria(d)
Base.in(d::Disjunction, c::StoppingCriterion) = false


## DISPLAY

function Base.show(io::IO, d::Disjunction)
    list = join(string.(_criteria(d)), ", ")
    print(io, "Disjunction($list)")
end

## RECURSION TO DEFINE `done`

# fallback for atomic criteria:
_done(criterion, state, old_done) = old_done || done(criterion, state)

# disjunction:
_done(d::Disjunction, state, old_done) =
    _done(d.a, state.a, _done(d.b, state.b, old_done))

done(d::Disjunction, state) = isnothing(state) ? false : _done(d, state, false)


## RECURSION TO BUILD MESSAGE

# fallback for atomic criteria:
function _message(criterion, state, old_message)
    done(criterion, state) && return old_message*message(criterion, state)
    return old_message
end

# disjunction:
_message(d::Disjunction, state, old_message) =
    _message(d.a, state.a,
             _message(d.b, state.b, old_message))

message(d::Disjunction, state) = _message(d, state, "")


## SYNTACTIC SUGAR

Base.zero(::Type{<:StoppingCriterion}) = Never()
+(a::StoppingCriterion, b::StoppingCriterion...) = Disjunction(a, b...)

