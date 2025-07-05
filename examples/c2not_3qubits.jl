using QuantumToolbox
using LinearAlgebra
using SparseArrays
using CairoMakie

# define the pulse functions
Omega_c(p, t) = if 0 <= t < p.T_c
        return p.omega_c / 2
    elseif p.T_c + 3 * p.T_t < t <= p.tmax
        return -p.omega_c / 2
    else
        return 0.0
    end

Omega_p(p,t) =
    if p.T_c<=t<p.T_c+p.T_t
        return p.omega_t/2
	elseif p.T_c + 2 * p.T_t <= t <= p.T_c + 3 * p.T_t
        return p.omega_t/2
    else
        return 0.0
	end


Omega_p2(p, t) = if p.T_c + p.T_t <= t <= p.T_c + 2 * p.T_t
        return -p.omega_t / 2
    else
        return 0.0
    end


function plot_pulses(params, pulse_funcs, t_plot)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Pulse Functions", xlabel = "Time", ylabel = "Amplitude")
    for pulse_func in pulse_funcs
        pulse_values = [pulse_func(params, t) for t in t_plot]
        lines!(ax, t_plot, pulse_values, label = string(pulse_func))
    end
    axislegend(ax; position = :rb, labelsize = 15)
    return fig
end

function plot_state_population(tlist, mc, state_labels)
    fig = Figure()
    ax = Axis(fig[1, 1], title = "State Population", xlabel = "Time", ylabel = "Fidelity")
    for (i, label) in enumerate(state_labels)
        lines!(ax, tlist, real(mc.expect[i, :]), label = label)
    end
    axislegend(ax; position = :rb, labelsize = 15)
    return fig
end

function get_mcsolve_solution(tlist, params)
    #control atom states
    gc = sparse(basis(2, 0)) #|0>
    rc = sparse(basis(2, 1)) #|1>

    #target atom states
    gt = sparse(basis(3, 0))
    et = sparse(basis(3, 1))
    rt = sparse(basis(3, 2))

    hc_op = gc * rc' + rc * gc'
    ht_op = rt * et' + et * rt'
    ht2_op = rt * gt' + gt * rt'
    hrr_op = rc * rc'
    hRR_op = rt * rt'

    Hc_op = tensor(hc_op, Qobj(qeye(2)), Qobj(qeye(3))) + tensor(Qobj(qeye(2)), hc_op, Qobj(qeye(3)))
    Ht_op1 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), ht_op)
    Ht_op2 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), ht2_op)

    HRr = params.V_dd .* (tensor(Qobj(qeye(2)), hrr_op, hRR_op) + tensor(hrr_op, Qobj(qeye(2)), hRR_op))
    HRR = params.Vdw .* tensor(hrr_op, hrr_op, Qobj(qeye(3)))

    L_c = sqrt(params.gamma_c) .* (gc * rc')
    L_R = sqrt(params.gamma_R) .* (gt * rt')

    h_L1 = -im / 2 .* (L_c' * L_c)
    h_L2 = -im / 2 .* (L_R' * L_R)

    H_L =
        tensor(h_L1, Qobj(qeye(2)), Qobj(qeye(3))) +
        tensor(Qobj(qeye(2)), h_L1, Qobj(qeye(3))) +
        tensor(Qobj(qeye(2)), Qobj(qeye(2)), h_L2)

    Hnhermitian = QobjEvo((H_L, HRr, HRR, (Hc_op, Omega_c), (Ht_op1, Omega_p), (Ht_op2, Omega_p2)))

    c_ops = [
        tensor(L_c, Qobj(qeye(2)), Qobj(qeye(3))) +
        tensor(Qobj(qeye(2)), L_c, Qobj(qeye(3))) +
        tensor(Qobj(qeye(2)), Qobj(qeye(2)), L_R),
    ]

    psi00 = tensor(gc, gc, gt)
    psi11 = tensor(gc, gc, et)
    psi_0 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), gt)
    psi_1 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), et)
    psi_2 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), rt)

    pulse_funcs = [Omega_p, Omega_p2, Omega_c]

    fig_pulses = plot_pulses(params, pulse_funcs, tlist)

    @time begin
    mc = mcsolve(
        Hnhermitian,
        psi00,
        tlist,
        c_ops;
        e_ops = [psi_0 * psi_0', psi_1 * psi_1', psi_2 * psi_2'],
        params = params,
    )
    end

    @time begin
    mc1 = mcsolve(
        Hnhermitian,
        psi11,
        tlist,
        c_ops;
        e_ops = [psi_0 * psi_0', psi_1 * psi_1', psi_2 * psi_2'],
        params = params,
    )
    end
    return fig_pulses, mc, mc1
end

function main(;omega_c, omega_t, V_dd, tau_c, tau_R, Vdw, T_t, T_c)
    params = (omega_c = omega_c, #MHz
              omega_t = omega_t, #MHz
              V_dd = V_dd, #MHz
              tau_c = tau_c,
              tau_R = tau_R,
              Vdw = Vdw, #MHz
              T_t = T_t,
              T_c = T_c,
              tmax = T_c * 2 + T_t * 3,
              gamma_c = 1 / tau_c,
              gamma_R = 1 / tau_R
    )

    tlist = LinRange(0, params.tmax, 1000)

    fig_pulses, mc, mc1 = get_mcsolve_solution(tlist, params)

    fig_pop1 = plot_state_population(tlist, mc, ["|g> Fidelity", "|e> Fidelity", "|r> Fidelity"])

    fig_pop2 = plot_state_population(tlist, mc1, ["|g> Fidelity", "|e> Fidelity", "|r> Fidelity"])

    return  fig_pulses, fig_pop1, fig_pop2
end


f1, f2, f3 = main(
    omega_c = 2 * π * 50, #MHz
    omega_t = 2 * π * 25, #MHz
    V_dd = 2 * π * 0, #MHz
    tau_c = 548,
    tau_R = 505,
    Vdw = 2 * π * 1, #MHz
    T_t = 1 * π / (2 * π * 25),
    T_c = 1 * π / (2 * π * 50),
)

f1
f2
save("mc.png", f2)
f3
save("mc1.png", f3)