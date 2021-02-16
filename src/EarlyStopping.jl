module EarlyStopping

using Dates
using Statistics

export StoppingCriterion,
    Never, NotANumber, TimeLimit, GL, Patience, UP, PQ, stopping_time

include("api.jl")
include("criteria.jl")
include("stopping_time.jl")

end # module
