using QuantumToolbox

omega_c = 2 * π * 50 #MHz
omega_t = 2 * π * 25 #MHz
T_t = π/omega_t
T_c = π/omega_c
tmax = T_c * 2 + T_t * 3
V_dd = 2 * π * 500 #MHz
tau_c = 548
tau_R = 505
gamma_c = 1/tau_c
gamma_R = 1/tau_R

nsteps = 1000
tlist = collect(range(0,tmax,length=nsteps))

#control atom states
gc = basis(2,0) #|0>
rc = basis(2,1) #|1>

#dagger
gc_dag = gc' # <0|
rc_dag = rc'  # <1|

#target atom states
gt = basis(3,0) 
et = basis(3,1)
rt = basis(3,2)

#dagger
gt_dag = gt'
et_dag = et'
rt_dag = rt'

#control field
function Omega_c(ps,t)
	if 0 <= t < ps.T_c
		return ps.omega_c/2
	elseif ps.T_c + 3 * ps.T_t < t <= ps.tmax
		return -ps.omega_c/2
	else
		return zero(ps.omega_c)/2
	end
end

function Omega_p(ps,t)
	if ps.T_c <= t < ps.T_c + ps.T_t
		return ps.omega_t/2
	elseif ps.T_c + 2 * ps.T_t <= t <= ps.T_c + 3 * ps.T_t
		return ps.omega_t/2
	else
		return zero(ps.omega_t)/2
	end
end

function Omega_p2(ps,t)
	if ps.T_c + ps.T_t <= t < ps.T_c + 2 * ps.T_t
		return -ps.omega_t/2
	else
		return zero(ps.omega_t)/2
	end
end

# omega_c
hc = (gc * rc' + rc * gc')

hrr = rc * rc'
hRR = rt * rt'

# omega p
ht = (rt * et' + et * rt')

ht2 = (rt * gt' + gt * rt')

L_c = sqrt(gamma_c) * (gc * rc')
L_R = sqrt(gamma_R) * (gt * rt')

function Hnhermitian(t)
    h1 = tensor(hc(t), eye(3)) + tensor(eye(2), ht(t))
    h2 = V_dd * tensor(hrr, hRR)
    h31 = -im/2 * (L_c' * L_c)
    h32 = -im/2 * (L_R' * L_R)
    h3 = tensor(h31, eye(3)) + tensor(eye(2), h32)
    return h1 + h2 + h3
end

h1 = tensor(hc,qeye(3))
h2 = tensor(qeye(2),ht)

H = QobjEvo(((h1,Omega_c),(h2,Omega_p)))


c_ops = [tensor(L_c, qeye(3)) + tensor(qeye(2), L_R)]

psi_0 = tensor(gc,gt)
psi_1 = tensor(gc,et)

params = (omega_c 
=omega_c, T_c = T_c, T_t=T_t, tmax=tmax, omega_t = omega_t)
# https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.QuantumObjectEvolution
mc = mcsolve(H, psi_0, tlist, c_ops; params = params, e_ops = [psi_0 * psi_0', psi_1 * psi_1'])