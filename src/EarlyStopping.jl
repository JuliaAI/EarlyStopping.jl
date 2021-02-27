module EarlyStopping

using Dates
using Statistics
import Base.+

export StoppingCriterion,
    Never, NotANumber, TimeLimit, GL, Patience, UP, PQ, NumberLimit,
    Disjunction, criteria, stopping_time, EarlyStopper,
    done!, message, needs_in_and_out_of_sample

include("api.jl")
include("criteria.jl")
include("disjunction.jl")
include("stopping_time.jl")
include("object_oriented_api.jl")

end # module
