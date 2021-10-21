module EarlyStopping

using Dates
using Statistics
import Base.+

export StoppingCriterion,
    Never,
    InvalidValue,
    NotANumber, # deprecated
    TimeLimit,
    GL,
    NumberSinceBest,
    Patience,
    UP,
    PQ,
    NumberLimit,
    Threshold,
    Disjunction,
    Warmup,
    criteria,
    stopping_time,
    EarlyStopper,
    done!,
    message,
    needs_training_losses,
    needs_loss

include("api.jl")
include("criteria.jl")
include("disjunction.jl")
include("object_oriented_api.jl")
include("stopping_time.jl")

end # module
