# EarlyStopping.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/EarlyStopping.jl/workflows/CI/badge.svg)](https://github.com/ablaom/EarlyStopping.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/EarlyStopping.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/EarlyStopping.jl?branch=master) |

A small package for applying early stopping criteria to
loss-generating iterative algorithms, with a view to applications to
training and optimization of machine learning models.

Includes the stopping criteria surveyed in [Prechelt, Lutz
(1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3):
"Early Stopping - But When?", in *Neural Networks: Tricks of the
Trade*, ed. G. Orr, Springer.

## Installation

```julia
using Pkg
Pkg.add("EarlyStopping")
```

## Sample usage

```julia
using EarlyStopping

stopper = EarlyStopper(Patience(2), NotANumber()) # muliple criteria
done!(stopper, 0.123) # false
done!(stopper, 0.234) # false
done!(stopper, 0.345) # true

julia> message(stopper)
"Early stop triggered by Patience(2) stopping criterion. "
```

The "object-oriented" interface demonstrated here is not optimized but
will suffice for the majority of use-cases. For performant code, use
the purely functional interface described under [Implementing new
criteria](#implementing-new-criteria) below.


## Criteria

To list all stopping criterion, do `subtypes(StoppingCriterion)`. Each
subtype `T` has a detailed doc-string queried with `?T` at the
REPL. Here is a short summary:


criterion             | description                                      | notation in Prechelt
----------------------|--------------------------------------------------|---------------------
`Never()`             | Never stop                                       | 
`NotANumber()`        | Stop when `NaN` encountered                       | 
`TimeLimit(t=0.5)`    | Stop after `t` in hours                          | 
`GL(alpha=2.0)`       | Stop after "Generalization Loss" exceeds `alpha` | ``GL_α``
`PQ(alpha=0.75, k=5)` | Stop after "Progress-modified GL" exceeds `alpha` | ``PQ_α``
`Patience(n=5)`       | Stop after `n` consecutive loss increases        | ``UP_s``
`Disjunction(c...)`   | Stop when any of the criteria `c` apply          | 


## Training losses

For criteria tracking both an "out-of-sample" loss and a "training"
loss (eg, stopping criterion of type `PQ`), specify `training=true` if
the update is for training, as in

    done!(stopper, 0.123, training=true)

In these cases, the out-of-sample update must always come after the
corresponding training update. Multiple training updates may precede
the out-of-sample update, as in the following example:

```julia
stopper = EarlyStopper(PQ(alpha=2.0, k=2))
done!(stopper, 9.5, training=true) # false
done!(stopper, 9.3, training=true) # false
done!(stopper, 10.0) # false

done!(stopper, 9.3, training=true) # false
done!(stopper, 9.1, training=true) # false
done!(stopper, 8.9, training=true) # false
done!(stopper, 8.0) # false

done!(stopper, 8.3, training=true) # false
done!(stopper, 8.4, training=true) # false
done!(stopper, 9.0) # true
```

## Stopping time

To determine the stopping time for a iterator `losses`, use
`stopping_time(criterion, losses)`. This is useful for testing new
criteria (see below). If the iterator terminates without a stop, `0`
is returned.

```julia
julia> stopping_time(NotANumber(), [10.0, 3.0, NaN, 4.0])
3

julia> stopping_time(NotANumber(), [10.0, 3.0, 5.0, 4.0])
0
```

If the lossses include training losses as described in [Training
losses](#training-losses) above, pass an extra `Bool` vector
`is_training`, as in

```julia
stopping_time(PQ(), 
              [0.123, 0.321, 0.52, 0.55, 0.56, 0.58],
              [true, true, false, true, true, false])
```


## Implementing new criteria

To implement a new stopping criterion, one must: 

- Define a new `struct` for the criterion, which must subtype
`StoppingCriterion`.

- Overload methods `update` and `done` for the new type.

- Optionally overload methods `message` and `update_training`.

We demonstrate this with a simplified version of the
[code](/src/criteria.jl) for `Patience`:


### Defining the new type

```julia
using EarlyStopping
import EarlyStopping: update, done, message

mutable struct Patience <: StoppingCriterion
    n::Int
end
Patience(; n=5) = Patience(n)
```

### Overloading `update` and `done`

All information to be "remembered" must passed around in an object
called `state` below, which is the return value of `update` (and
`update_training`). The `update` function has two methods - one for
initialization, without a `state` argument, and one for all subsequent
loss updates, which does:

```julia
update(criterion::Patience, loss) = (loss=loss, n_increases=0) # state

function update(criterion::Patience, loss, state)
    old_loss, n = state
    if loss > old_loss
        n += 1
    else
        n = 0
    end
    return (loss=loss, n_increases=n) # state
end
```

The `done` method returns `true` or `false` depending on the `state`:

```julia
done(criterion::Patience, state) = state.n_increases == criterion.n
```

### Optionally overload `message` and `training_update`

The final message of an `EarlyStopper` is generated by a `message`
method for `StoppingCriterion`. Here is the fallback (which does not
use `state`):

```julia
message(criteria::StoppingCriterion, state)  = "Early stop triggered by "*
    "$criterion stopping criterion. "
```

The optional `update_training` methods (two for each criterion) have
the same signature as the `update` methods above. Refer to the `PQ`
[code](/src/criteria) for an example.

