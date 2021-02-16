module EarlyStopping

using Dates
using Statistics
import Base.+

export StoppingCriterion,
    Never, NotANumber, TimeLimit, GL, Patience, UP, PQ,
    Disjunction, criteria, stopping_time

include("api.jl")
include("criteria.jl")
include("disjunction.jl")
include("stopping_time.jl")

end # module
