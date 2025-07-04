using QuantumToolbox
using LinearAlgebra
using SparseArrays

omega_c = 2 * π * 50 #MHz
omega_t = 2 * π * 25 #MHz
T_t = 1 * π/omega_t
T_c = 1 * π/omega_c
tmax = T_c * 2 + T_t * 3
V_dd = 2 * π * 0 #MHz
tau_c = 548
tau_R = 505
gamma_c = 1/tau_c
gamma_R = 1/tau_R
Vdw = 2 * π * 1 #MHz


tlist = LinRange(0, tmax, 1000)

#control atom states
gc = sparse(basis(2,0)) #|0>
rc = sparse(basis(2,1)) #|1>

#target atom states
gt = sparse(basis(3,0))
et = sparse(basis(3,1))
rt = sparse(basis(3,2))

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


function Omega_p2(p,t)
    if p.T_c + p.T_t <= t <= p.T_c + 2 * p.T_t
        return -p.omega_t / 2
    else
        return 0.0
    end
end

cur_p = (omega_c=omega_c, omega_t=omega_t, T_c=T_c, T_t=T_t, tmax=tmax)
t_plot = LinRange(0, tmax, 1000)
pulse_valuesp = [Omega_p(cur_p, t) for t in t_plot]
pulse_valuesp2 = [Omega_p2(cur_p, t) for t in t_plot]
pulse_valuesc = [Omega_c(cur_p, t) for t in t_plot]
using CairoMakie


plot(t_plot, pulse_valuesp)
plot(t_plot, pulse_valuesp2)
plot(t_plot, pulse_valuesc)


xlabel("Time")
ylabel("Amplitude")
title("Pulse Function and its Integral")


hc_op = gc * rc' + rc * gc'
ht_op = rt * et' + et * rt'
ht2_op = rt * gt' + gt * rt'
hrr_op = rc * rc'
hRR_op = rt * rt'

Hc_op = tensor(hc_op, Qobj(qeye(2)), Qobj(qeye(3))) + tensor(Qobj(qeye(2)),hc_op,  Qobj(qeye(3)))
Ht_op1 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), ht_op)
Ht_op2 = tensor(Qobj(qeye(2)), Qobj(qeye(2)), ht2_op)

HRr = V_dd .* (tensor(Qobj(qeye(2)), hrr_op, hRR_op) + tensor(hrr_op, Qobj(qeye(2)), hRR_op))
HRR = Vdw .*  tensor(hrr_op, hrr_op, Qobj(qeye(3)))

L_c = sqrt(gamma_c) .* (gc * rc')
L_R = sqrt(gamma_R) .* (gt * rt')

h_L1 = -im/2 .* (L_c' * L_c)
h_L2 = -im/2 .* (L_R' * L_R)

H_L = tensor(h_L1, Qobj(qeye(2)), Qobj(qeye(3))) + tensor(Qobj(qeye(2)), h_L1, Qobj(qeye(3))) + tensor(Qobj(qeye(2)), Qobj(qeye(2)), h_L2)

Hnhermitian = QobjEvo((H_L, HRr, HRR, (Hc_op,Omega_c), (Ht_op1, Omega_p), (Ht_op2, Omega_p2)))

c_ops = [tensor(L_c, Qobj(qeye(2)), Qobj(qeye(3))) + tensor(Qobj(qeye(2)), L_c, Qobj(qeye(3))) + tensor(Qobj(qeye(2)), Qobj(qeye(2)), L_R)]


psi00 = tensor(gc,gc,gt)
psi11 = tensor(gc,gc,et)
psi_0 = tensor(Qobj(qeye(2)),Qobj(qeye(2)),gt)
psi_1 = tensor(Qobj(qeye(2)),Qobj(qeye(2)),et)
psi_2 = tensor(Qobj(qeye(2)),Qobj(qeye(2)),rt)

mc = mcsolve(
    Hnhermitian,
    psi00,
    tlist,
    c_ops;
    e_ops = [psi_0 * psi_0', psi_1 * psi_1', psi_2 * psi_2'],
    params = (T_c = T_c, T_t = T_t, tmax = tmax, omega_c = omega_c, omega_t = omega_t),
)