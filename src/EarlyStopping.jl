module EarlyStopping

using Dates

export StoppingCriterion,
    Never, NotANumber, TimeLimit, GL, Patience, UP, stopping_time

include("api.jl")
include("criteria.jl")
include("stopping_time.jl")

end # module
