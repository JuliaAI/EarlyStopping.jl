# EarlyStopping.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/EarlyStopping.jl.jl/workflows/CI/badge.svg)](https://github.com/ablaom/EarlyStopping.jl.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/EarlyStopping.jl.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/EarlyStopping.jl.jl?branch=master) |

A small package for applying early stopping criteria to
loss-generating iterative algorithms, in particular, machine learning
training and optimization algorithms.

# Includes the stopping criterion surveyed in [Prechelt, Lutz (1998):
# "Early Stopping - But When?", in *Neural Networks: Tricks of the
# Trade*, ed. G. Orr, Springer.](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3)

## Installation

```julia
using Pkg
Pkg.add("EarlyStopping")
```

## Sample usage

```julia
stopper = EarlyStopper(Patience(1), NotANumber())
done!(stopper, 0.123) # false
done!(stopper, 0.234) # true

julia> message(stopper)
"Early stop triggered by Patience(1) stopping criterion. "
```

The "object-oriented" interface demonstrated here is not optimized but
will suffice for the majority of use-cases. For performant code, use
the purely functional interface described under [Implementing new
criteria](#implementing-new-criteria) below.


## Built-in criteria

Each `StoppingCriterion` type listed below has a detailed doc-string;
do `?ChosenCriteria` for details on a given type.




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



# criterion                     | notation in Prechelt
# ------------------------------|--------------------------------
# `Never`                       | -
# `NotANumber`                  | -
# `TimeLimit`                   | -
# `GL`                          | ``GL_α``
# `PQ`                          | ``PQ_α``
# `Patience`                    | ``UP_s``

