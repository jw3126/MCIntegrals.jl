module MCIntegrals
using Requires

include("rng.jl")
include("core.jl")
include("vegas.jl")
include("cuba.jl")
include("plots.jl")

function __init__()
    @require HCubature="19dc6840-f33b-545b-b366-655c7e3ffd49" include("hcubature.jl")
end

end # module
