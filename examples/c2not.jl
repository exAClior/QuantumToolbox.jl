using Pkg; Pkg.activate("examples")
using QuantumToolbox, LinearAlgebra, SparseArrays
using Optim
# import Mooncake
# import Enzyme
import Zygote
using SciMLSensitivity
import DifferentiationInterface as DI

Omega_c_er(p, t) = p.cer_a1 * exp(-(t - p.cer_b1)^2 / (2 * p.cer_c1^2)) + p.cer_a2 * exp(-(t - p.cer_b2)^2 / (2 * p.cer_c2^2))

Omega_c_gr(p, t) = p.cgr_a1 * exp(-(t - p.cgr_b1)^2 / (2 * p.cgr_c1^2)) + p.cgr_a2 * exp(-(t - p.cgr_b2)^2 / (2 * p.cgr_c2^2))

Omega_p_AR(p, t) = p.ar_a1 * exp(-(t - p.ar_b1)^2 / (2 * p.ar_c1^2)) + p.ar_a2 * exp(-(t - p.ar_b2)^2 / (2 * p.ar_c2^2))

Omega_p_BR(p, t) = p.br_a1 * exp(-(t - p.br_b1)^2 / (2 * p.br_c1^2)) + p.br_a2 * exp(-(t - p.br_b2)^2 / (2 * p.br_c2^2))

function make_system_params(omega_r, omega_max, Dta, V_dd, V_vdws, tau_c, tau_p, tau_R)
    T_p = 16 * π * Dta / (3 * omega_max^2) #target pulse duration
    T_c = π / omega_r #control pulse duration
    tmax = T_c * 2 + T_p #total time of the simulation
    return (
        gamma_c=1 / tau_c,
        gamma_p=1 / tau_p,
        gamma_R=1 / tau_R,
        Dta=Dta,
        V_dd=V_dd,
        V_vdws=V_vdws,
        tmax=tmax,
    )
end

# https://discourse.julialang.org/t/sensitivity-of-an-odeproblem-defined-by-another-package-quantumtoolbox/129654/5?u=exaclior
function make_loss(Hnhermitian, tlist, c_ops, init_states, out_states)
    # https://discourse.julialang.org/t/automatic-differentiation-of-quantum-master-equation-using-zygote/127715/3
    function loss(pulse_params)
        mcs = [mcsolve(Hnhermitian, init_state, tlist, c_ops; e_ops=[out_state * out_state'], progress_bar=Val(false), sensealg=BacksolveAdjoint(autojacvec=
            EnzymeVJP()), params=pulse_params) for (init_state, out_state) in zip(init_states, out_states)]

        return sum(x -> (1 - real(x.expect[1, end]))^2, mcs)
    end
end

function make_ops(system_params)
    g = sparse(basis(3, 0)) #|0>  
    e = sparse(basis(3, 1)) #|1>   
    r = sparse(basis(3, 2)) #|r>

    #target atom states
    A = sparse(basis(3, 0))
    B = sparse(basis(3, 1))
    R = sparse(basis(3, 2))

    id3 = Qobj(qeye(3))

    hc_op1 = e * r' + r * e'
    hc_op2 = g * r' + r * g'

    Hc_op1 = tensor(hc_op1, id3, id3) + tensor(id3, hc_op1, id3)
    Hc_op2 = tensor(hc_op2, id3, id3) + tensor(id3, hc_op2, id3)
    # target Hamiltonian
    ht_op1 = A * R' + R * A'
    ht_op2 = B * R' + R * B'

    Ht_op1 = tensor(id3, id3, ht_op1)
    Ht_op2 = tensor(id3, id3, ht_op2)

    ht_dta = -system_params.Dta .* R * R'
    Ht_dta = tensor(id3, id3, ht_dta)
    # interaction Hamiltonian
    hrr_op = r * r'
    hRR_op = R * R'
    HRr1 = system_params.V_dd .* tensor(hrr_op, id3, hRR_op)
    HRr2 = system_params.V_dd .* tensor(id3, hrr_op, hRR_op)
    HRr = HRr1 + HRr2

    L_ce = sqrt(system_params.gamma_c) .* sparse(e * r')
    L_cg = sqrt(system_params.gamma_c) .* sparse(g * r')
    L_P = sqrt(system_params.gamma_R) .* sparse(A * R')
    L_R = sqrt(system_params.gamma_R) .* sparse(B * R')

    Hnhermitian =
        QobjEvo((HRr, Ht_dta, (Hc_op1, Omega_c_er), (Hc_op2, Omega_c_gr), (Ht_op1, Omega_p_AR), (Ht_op2, Omega_p_BR)))

    c_ops = [tensor((L_cg + L_ce), id3, id3) + tensor(id3, (L_cg + L_ce), id3) + tensor(id3, id3, (L_R + L_P))]

    init_11 = tensor(g, g, A) #|00>_c |0>_t  #initial input state
    out_11 = tensor(g, g, A) #|00>_c |1>_t  #expected output state

    init_12 = tensor(g, g, B) #|00>_c |0>_t  #initial input state
    out_12 = tensor(g, g, B) #|00>_c |1>_t  #expected output state

    init_21 = tensor(g, e, A) #|01>_c |0>_t  #initial input state
    out_21 = tensor(g, e, B) #|01>_c |1>_t  #expected output state

    init_22 = tensor(g, e, B) #|01>_c |0>_t  #initial input state
    out_22 = tensor(g, e, A) #|01>_c |1>_t  #expected output state

    init_31 = tensor(e, g, A) #|01>_c |0>_t  #initial input state
    out_31 = tensor(e, g, B) #|01>_c |1>_t  #expected output state

    init_32 = tensor(e, g, B) #|01>_c |0>_t  #initial input state
    out_32 = tensor(e, g, A) #|01>_c |1>_t  #expected output state

    init_41 = tensor(e, e, A) #|11>_c |0>_t  #initial input state
    out_41 = tensor(e, e, A) #|11>_c |1>_t  #expected output state

    init_42 = tensor(e, e, B) #|11>_c |0>_t  #initial input state
    out_42 = tensor(e, e, B) #|11>_c |1>_t  #expected output state
    return Hnhermitian, c_ops, (
        init_11, out_11,
        init_12, out_12,
        init_21, out_21,
        init_22, out_22,
        init_31, out_31,
        init_32, out_32,
        init_41, out_41,
        init_42, out_42
    ) 
end

function main(; omega_r, omega_max, Dta, V_dd, V_vdws, tau_c, tau_p, tau_R, nsteps)

    system_params = make_system_params(omega_r, omega_max, Dta, V_dd, V_vdws, tau_c, tau_p, tau_R)

    tlist = LinRange(0.0, system_params.tmax,  nsteps)

    Hnhermitian, c_ops, (
        init_11, out_11,
        init_12, out_12,
        init_21, out_21,
        init_22, out_22,
        init_31, out_31,
        init_32, out_32,
        init_41, out_41,
        init_42, out_42
    ) = make_ops(system_params)

    pulse_params = (
        cer_a1=0.5, cer_b1=0.0, cer_c1=0.1,
        cer_a2=0.5, cer_b2=0.0, cer_c2=0.1,
        cgr_a1=0.5, cgr_b1=0.0, cgr_c1=0.1,
        cgr_a2=0.5, cgr_b2=0.0, cgr_c2=0.1,
        ar_a1=0.5, ar_b1=0.0, ar_c1=0.1,
        ar_a2=0.5, ar_b2=0.0, ar_c2=0.1,
        br_a1=0.5, br_b1=0.0, br_c1=0.1,
        br_a2=0.5, br_b2=0.0, br_c2=0.1
    )

    f = make_loss(Hnhermitian, tlist, c_ops, [
            init_11, init_12, init_21, init_22,
            init_31, init_32, init_41, init_42
        ], [
            out_11, out_12, out_21, out_22,
            out_31, out_32, out_41, out_42
        ]
    )

    return Zygote.gradient(f, pulse_params)  # fast, use this
    # return DI.gradient(f, backend, pulse_params)  # slow, do not do this
end

main(;
    omega_r=2 * π * 50, #MHz
    omega_max=2 * π * 50,
    Dta=2 * π * 1200, #MHz #dutuning of the intermediate state
    V_dd=2 * π * 500, #MHz #dipole-dipole interaction strength
    V_vdws=2 * π * 1, #MHz #van der Waals interaction strength
    tau_c=548, #mu s  #control qubit relaxation time
    tau_p=0.131, #mu s #target qubit relaxation time(intermediate state)
    tau_R=505, #mu s #target qubit relaxation time (Rydberg state)
    nsteps=1000
)

