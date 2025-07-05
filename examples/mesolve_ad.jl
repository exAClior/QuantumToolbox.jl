using Pkg; Pkg.activate("examples")
using QuantumToolbox, LinearAlgebra, SparseArrays
using Optim
import Zygote
using SciMLSensitivity
import DifferentiationInterface as DI

Omega_c_er(p, t) = p.cer_a1 * exp(-(t - p.cer_b1)^2 / (2 * p.cer_c1^2)) + p.cer_a2 * exp(-(t - p.cer_b2)^2 / (2 * p.cer_c2^2))

function loss(pulse_params)
    mc = mcsolve(
            Hnhermitian,
            init_state,
            tlist,
            c_ops;
            e_ops = [out_state * out_state'],
            progress_bar = Val(false),
            sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
            params = pulse_params,)
	return (1- real(mc.expect[1,end])^2)
end

H0 = sparse(basis(3,1) * basis(3,0)' + basis(3,0) * basis(3,1)')

Hc_op1 = sparse(basis(3,2) * basis(3,0)' + basis(3,0) * basis(3,2)')

Hnhermitian = QobjEvo((H0, (Hc_op1, Omega_c_er)))

init_state = sparse(basis(3,0))

out_state = sparse(basis(3,1))

params = (cer_a1 = 0.1, cer_b1 = 0.0, cer_c1 = 0.1,
          cer_a2 = 0.5, cer_b2 = 0.0, cer_c2 = 0.2)
tlist = LinRange(0.0, 1.0, 1000)

c_ops = [(0.1 .* sparse(basis(3, 1) * basis(3, 0)') + 0.2 .* sparse(basis(3, 2) * basis(3, 0)'))]

loss(params)

# https://docs.sciml.ai/SciMLStructures/stable/example/
Zygote.gradient(loss,params)


optimize(loss,)


