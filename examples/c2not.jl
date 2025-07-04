using QuantumToolbox
using LinearAlgebra
using SparseArrays

omega_r = 2 * π * 50 #MHz
omega_max = omega_r
omega_R = 2.5 * omega_r #MHz #Applying on the target qubit
omega_c = 2 * π * 50 #MHz #Applying on the control qubit
Dta = 2 * π * 1200 #MHz #dutuning of the intermediate state
T_p = 16 * π * Dta/(3 * omega_max^2) #target pulse duration
T_c = π/omega_r #control pulse duration
tmax = T_c * 2 + T_p #total time of the simulation
V_dd = 2 * π * 500 #MHz #dipole-dipole interaction strength
V_vdws = 2 * π * 1 #MHz #van der Waals interaction strength
tau_c = 548 #mu s  #control qubit relaxation time
tau_p = 0.131 #mu s #target qubit relaxation time(intermediate state)
tau_R = 505 #mu s #target qubit relaxation time (Rydberg state)
gamma_c = 1/tau_c # control qubit decay rate
gamma_p = 1/tau_p # target qubit decay rate (intermediate state)
gamma_R = 1/tau_R # target qubit decay rate (Rydberg state)


len_mult = 1
nsteps = 1000
tlist = LinRange(0.0, len_mult*tmax, len_mult*nsteps)

g = basis(3, 0) #|0>  
e = basis(3, 1) #|1>   
r = basis(3, 2) #|r>

#dagger
g_dag = g'  # <0|
e_dag = e'  # <1|
r_dag = r'  # <r|

#target atom states
A = basis(3, 0)
B = basis(3, 1)
R = basis(3, 2)


id3 = Qobj(sparse(I,3,3)) 
id4 = Qobj(sparse(I,4,4))

hc_op1 = sparse(e * r' + r * e')
hc_op2 = sparse(g * r' + r * g')

Hc_op1 = tensor(hc_op1, id3, id3) + tensor(id3, hc_op1, id3)
Hc_op2 = tensor(hc_op2, id3, id3) + tensor(id3, hc_op2, id3) 
# target Hamiltonian
ht_op1 = sparse(A * R' + R * A')
ht_op2 = sparse(B * R' + R * B')

Ht_op1 = tensor(id3, id3, ht_op1) 
Ht_op2 = tensor(id3, id3, ht_op2) 

ht_dta = sparse(- Dta * R * R')
Ht_dta = tensor(id3, id3, ht_dta)
# interaction Hamiltonian
hrr_op = sparse(r * r')
hRR_op = sparse(R * R')
HRr1 =  V_dd * tensor(hrr_op, id3, hRR_op)
HRr2 =  V_dd * tensor(id3, hrr_op, hRR_op)
HRr = HRr1 + HRr2   

L_ce = sqrt(gamma_c) .* sparse(e * r')
L_cg = sqrt(gamma_c) .* sparse(g * r')
L_P = sqrt(gamma_R) .* sparse(A * R')
L_R = sqrt(gamma_R) .* sparse(B * R')

Omega_c_er(p, t) = p.a1 * exp(-(t - p.b1)^2 / (2 * p.c1^2)) + p.a2 * exp(-(t - p.b2)^2 / (2 * p.c2^2))

Omega_c_gr(p, t) = p.a1 * exp(-(t - p.b1)^2 / (2 * p.c1^2)) + p.a2 * exp(-(t - p.b2)^2 / (2 * p.c2^2))


Omega_p_AR(p, t) = p.a1 * exp(-(t - p.b1)^2 / (2 * p.c1^2)) + p.a2 * exp(-(t - p.b2)^2 / (2 * p.c2^2))

Omega_p_BR(p, t) = p.a1 * exp(-(t - p.b1)^2 / (2 * p.c1^2)) + p.a2 * exp(-(t - p.b2)^2 / (2 * p.c2^2))

Hnhermitian =
    QobjEvo((HRr, Ht_dta, (Hc_op1, Omega_c_er), (Hc_op2, Omega_c_gr), (Ht_op1, Omega_p_AR), (Ht_op2, Omega_p_BR)))


c_ops = [tensor((L_cg + L_ce), id3, id3) + tensor(id3, (L_cg + L_ce), id3) + tensor(id3, id3, (L_R + L_P))]


init_11 = sparse(tensor(g, g, A)) #|00>_c |0>_t  #initial input state
out_11 = sparse(tensor(g, g, A)) #|00>_c |1>_t  #expected output state

init_12 = sparse(tensor(g, g, B)) #|00>_c |0>_t  #initial input state
out_12 = sparse(tensor(g, g, B)) #|00>_c |1>_t  #expected output state

init_21 = sparse(tensor(g, e, A)) #|01>_c |0>_t  #initial input state
out_21 = sparse(tensor(g, e, B)) #|01>_c |1>_t  #expected output state

init_22 = sparse(tensor(g, e, B)) #|01>_c |0>_t  #initial input state
out_22 = sparse(tensor(g, e, A)) #|01>_c |1>_t  #expected output state

init_31 = sparse(tensor(e, g, A)) #|01>_c |0>_t  #initial input state
out_31 = sparse(tensor(e, g, B)) #|01>_c |1>_t  #expected output state

init_32 = sparse(tensor(e, g, B)) #|01>_c |0>_t  #initial input state
out_32 = sparse(tensor(e, g, A)) #|01>_c |1>_t  #expected output state

init_41 = sparse(tensor(e, e, A)) #|11>_c |0>_t  #initial input state
out_41 = sparse(tensor(e, e, A)) #|11>_c |1>_t  #expected output state

init_42 = sparse(tensor(e, e, B)) #|11>_c |0>_t  #initial input state
out_42 = sparse(tensor(e, e, B)) #|11>_c |1>_t  #expected output state

# https://discourse.julialang.org/t/automatic-differentiation-of-quantum-master-equation-using-zygote/127715/3
mc = mcsolve(Hnhermitian, init_11, tlist, c_ops; e_ops = [init_11 * init_11', out_11 * out_11'],params=(a1=0.2,a2=0.3,b1=0.3,c1=2,b2=1.4,c2=3))