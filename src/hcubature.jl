import HCubature
export HCubatureAlg

struct HCubatureAlg{K}
    kw::K
end

function HCubatureAlg(;kw...)
    HCubatureAlg(kw)
end

function integral_kernel(f, dom::Domain, alg::HCubatureAlg)
    value, err = HCubature.hcubature(f, dom.lower, dom.upper; alg.kw...)
    (value=value, std=err) # TODO err is not really a standard deviation, what should be done?
end
