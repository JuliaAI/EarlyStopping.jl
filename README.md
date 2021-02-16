# EarlyStopping.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/EarlyStopping.jl.jl/workflows/CI/badge.svg)](https://github.com/ablaom/EarlyStopping.jl.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/EarlyStopping.jl.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/EarlyStopping.jl.jl?branch=master) |

# Includes stopping criterion surveyed in [Prechelt, Lutz (1998):
# "Early Stopping - But When?", in *Neural Networks: Tricks of the
# Trade*, ed. G. Orr, Springer.](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3)


# criterion                     | notation in Prechelt
# ------------------------------|--------------------------------
# `Never`                       | -
# `NotANumber`                  | -
# `TimeLimit`                   | -
# `GL`                          | ``GL_α``
# `PQ`                          | ``PQ_α``
# `Patience`                    | ``UP_s``

