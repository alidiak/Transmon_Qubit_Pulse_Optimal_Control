#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:23:27 2021

@author: Alex Lidiak  
"""
#imports
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations
from itertools import product
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint

#Function definitions 
def zero_mat(N):#Generates matrix of zeros
    zero_gate = np.array([[0,0],[0,0]])
    init = np.array([[0,0],[0,0]])
    if N < 2:
        return 1
    for i in range(0,N - 1):
        zero_gate = torch.tensor(np.kron(zero_gate,init))
    return zero_gate

def sum_pauli(coef, gate):#Sums Pauli gates with coefficients 
    N = len(coef)#number of qubits
    total_pauli = zero_mat(N)
    #Summing all Z gates
    for i in range(0,N):
        pauli_temp = 1
        for j in range(0,i):
            pauli_temp = torch.tensor(np.kron(pauli_temp,id))
        pauli_temp = torch.tensor(np.kron(pauli_temp, gate))
        for j in range(i+1,N):
            pauli_temp = torch.tensor(np.kron(pauli_temp,id))
        total_pauli = total_pauli + coef[i]*pauli_temp
    return total_pauli

class Applied_Hamiltonian(nn.Module): # inherits from nn.Module just for
    # compatibility with torchdiffeq.odeint
    """

    Parameters
    ----------
    t : float - Time at which to evaluate Omega(t)
    A : Tensor of Gaussian Pulse Amplitudes (to be trained)
    T : torch.float - Total evolution time during which the pulses are applied
    gate_list: list of length n_gates - the list of gates to multiply omega(t) by

    Functions
    -------
    Get_H1_t : 
    returns two torch.double matrices Hr_t = Re(H1_t) & Hi_t = Im(H1_t)
    I.e. the time dependent Hamiltonian H1_t split into the real and imag 
    components for each of the respective real/imaginary schrodinger equations.
    
    Input : t - time to be evalutated at
    ---
    
    -----
    Schrodinger_eq : function meant to be integrated via an ODE solver
    returns: dU - derivative of the schrodinger equation split into [real, imag]
    which are concatenated together.
    
    Input :
    t: Tensor of times at which to evaluate
    U: [real, imag] x [Hilbert_space x Hilbert_space] tensor, 
        tensor - U[0, :, :] - corresponds to the real component and 
        tensor - U[1,:,:] - corresponds to the imag component. 
        (same as for output dU above)
    -----

    """
    def __init__(self, A, T, gates, H0):
        super(Applied_Hamiltonian, self).__init__()
        self.A = nn.Parameter(A) # Parameter to keep gradient for
        self.T = T
        self.gates = gates
        self.H0 = H0 # static Hamiltonian to be applied

    def Get_H1_t(self, t):
        M, N2 = self.A.shape
        n_gates = len(self.gates)
        N = int(N2/n_gates)
        a = torch.tensor(0.5*(self.T/M), dtype = self.A.dtype)
        
        H1t = torch.zeros((2**N, 2**N), dtype = torch.cdouble)
        for m in range(M):
            tm = torch.tensor(m*(self.T/M), dtype = self.A.dtype)
            for gg in range(len(self.gates)): 
                # Get the N coefficients for each gate application
                coef = self.A[m, gg*N:N*(gg+1)]*torch.exp(-(t-tm).square()/a.square())
                
                # Use the gate sum function to sum these contributions applied to given gates
                gate_contrib = sum_pauli(coef, self.gates[gg])
                
                # Finally, accumulate these contributions in the time dependent Hamiltonian
                H1t += gate_contrib
                
        Hr_t = H1t.real
        Hi_t = H1t.imag
        
        return Hr_t, Hi_t
    
    def Schrodinger_eq(self, t, U):
        
        # Getting the real and imaginary Unitary components
        Ur = U[0,...]
        Ui = U[1,...]
        
        # Getting the real and imaginary Applied Hamiltonian peices    
        Hr, Hi = self.Get_H1_t(t)
        
        # Implementing the Real Schrodinger Eq.
        dUr = self.H0@Ui + Hr@Ui + Hi@Ur
                
        # Implementing the Imag Schrodinger Eq.
        dUi = Hi@Ui - self.H0@Ur - Hr@Ur
        
        # combine back into a single U
        dU = torch.cat((dUr[None,...], dUi[None,...]), dim=0) 
        
        return dU

# N = 3
# n_t = 11
# T = 3
# M = 12
# sx = np.array([[0, 1], [1, 0]])
# sy = np.array([[0,-1j],[1j,0]])
# gates = [sx, sy]
# H0 = torch.diag(torch.tensor([6,0,0,-2,0,-2,-2,0], dtype=torch.double))

# A = torch.rand([M, len(gates)*N], dtype=torch.double)*2*np.pi
# # Initialize an instance of the Applied Hamiltonian class
# this_H1t = Applied_Hamiltonian(A, T, gates, H0)

# U0r = torch.eye(2**N, dtype = torch.double) # real comp of U0
# U0i = torch.zeros((2**N,2**N), dtype = torch.double) # imag comp of U0
# U0 = torch.cat((U0r[None,...], U0i[None,...]), dim=0) # combining them into a single I.C. 

# torchpi = torch.tensor(np.pi, dtype = torch.double)
# t_list = torch.linspace(0, torchpi, n_t)
# UT = odeint(this_H1t.Schrodinger_eq, U0, t_list)
    
# %%

def fidelity_ml(J, B, M, target_gate, t, N_iter):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Mon Aug 16 4:33 2021

    @author: Alex Lidiak & Bora Basyildiz
    """

    #Pauli Matricies 
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1, 0], [0, -1]])
    id = np.array([[1,0],[0,1]])

    #variable initializations
    N = len(B)
    torch.manual_seed(9)
    infidelity_list=torch.zeros([N_iter,1])

    #J coefficients gathering (only if J is in N x N matrix, otherwise set J_coef=J)
    J_coef = []
    for i in range(0,len(J) - 1):
        for j in range(0,len(J) - i - 1):
            J_coef.append(J[i,j].item())

    #H0 generation
    permuts = [1,1]
    for i in range(2,N):
        permuts.append(0)
    permuts = list(set(permutations(permuts,N)))
    permuts.sort()
    permuts.reverse()#All permutations of ZZ coupling stored as bit arrays
    H0 = zero_mat(N)
    for i,u in enumerate(permuts):#summing ZZ permutations and J constants
        ZZ_temp = 1
        for p in u:
            if p==1:
                ZZ_temp = torch.tensor(np.kron(ZZ_temp,sz))
            else:
                ZZ_temp = torch.tensor(np.kron(ZZ_temp,id))
        H0 = H0 + J_coef[i]*ZZ_temp

    H0 = H0 + sum_pauli(B,sz)

    #Unitary group generation
    SU = []
    pauli_int = [1, 2, 3, 4]#eq to [sx,sy,sz,id]
    perms = list(product(pauli_int,repeat=N))#all permutations of paulis
    for p in perms:#mapping integers to pauli 
        unitary = 1
        for pauli in p:
            if pauli == 1:
                unitary = torch.tensor(np.kron(unitary,sx),dtype=torch.cdouble)
            elif pauli == 2:
                unitary = torch.tensor(np.kron(unitary,sy),dtype=torch.cdouble)
            elif pauli == 3:
                unitary = torch.tensor(np.kron(unitary,sz),dtype=torch.cdouble)
            elif pauli == 4:
                unitary = torch.tensor(np.kron(unitary,id),dtype=torch.cdouble)
        SU.append(unitary)

    #### DEFINE/INPUT the gates you'd like Gaussian pulses to be applied on ####
    gates = [sx, sy]

    #These are the coefficients we are optimizing
    R = torch.rand([M, len(gates)*N], dtype=torch.double) *2*np.pi # Random initialization (between 0 and 2pi)
    # R.requires_grad = True # set flag so we can backpropagate
    # in new formulation, the nn.Param call in Applied_Hamiltonian will require gradients
    
    t = torch.tensor(t)
    
    # Initialize an instance of the Applied Hamiltonian class (to be optimized)
    H1t = Applied_Hamiltonian(R, t, gates, torch.tensor(H0, dtype=torch.double))

    # set the list of times to evaluate the ODE at (doing minimal # of t_evals)
    t_list = torch.linspace(0, t, 2)

    #Optimizer settings(can be changed & optimized)
    # lr=0.3 #learning rate
    lr = 3

    opt = 'SGD'  # Choose optimizer - ADAM, SGD (typical). ADAMW, ADAMax, Adadelta,  
                        # Adagrad, Rprop, RMSprop, ASGD, also valid options.     
    sched = 'Plateau'  # Choose learning rate scheduler - Plateau, Exponential (typical), Step
    
    if opt=='ADAM': optimizer = torch.optim.Adam([H1t.A], lr = lr, weight_decay=1e-6)
    elif opt=='ADAMW': optimizer = torch.optim.AdamW([H1t.A], lr = lr, weight_decay=0.01)
    elif opt=='ADAMax': optimizer = torch.optim.Adamax([H1t.A], lr = lr, weight_decay=0.01)
    elif opt=='RMSprop': optimizer = torch.optim.RMSprop([H1t.A], lr = lr, momentum=0.2)
    elif opt=='Rprop': optimizer = torch.optim.Rprop([H1t.A], lr = lr)
    elif opt=='Adadelta': optimizer = torch.optim.Adadelta([H1t.A], lr = lr) 
    elif opt=='Adagrad': optimizer = torch.optim.Adagrad([H1t.A], lr = lr)
    elif opt=='SGD': optimizer = torch.optim.SGD([H1t.A], lr = lr, momentum=0.99, nesterov=True)
    elif opt=='ASGD': optimizer = torch.optim.ASGD([H1t.A], lr = lr)
    else: optimizer=None; opt='None'
        
    if sched=='Step': scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=N_iter/10, gamma=0.9)
    elif sched=='Exponential': scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    elif sched=='Plateau': scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.03, factor=0.3 , patience= 20 ); loss_in=True; 
    else: scheduler=None; sched='None'

    for n in range(0,N_iter):
        #Creating Hamilontian
        # U_Exp = 1
        # for i in range(0,N):
        #     U_Exp = torch.tensor(np.kron(U_Exp,id),dtype=dt) #initializing unitary
        # for m in range(0,M):#Product of pulses
            
        #     pulse_coef = R[m]
        #     H1 = sum_pauli(pulse_coef[:N],sx) + sum_pauli(pulse_coef[N:],sy)
            
        #     U_Exp = torch.matmul(torch.matrix_exp(-1j*(H0+H1)*t/M),U_Exp)
        
        # !!! (ALERT) : This is the new added ODE solver, added by Alex Lidiak on
        # Dec. 17 2021. Meant to replace the full m loop previously implemented above
        # initializing unitary 
        U0r = torch.eye(2**N, dtype = torch.double) # real comp of U0
        U0i = torch.zeros((2**N,2**N), dtype = torch.double) # imag comp of U0
        U0 = torch.cat((U0r[None,...],U0i[None,...]), dim=0) # combining them into a single I.C. 
                
        U_tlist = odeint(H1t.Schrodinger_eq, U0, t_list)  
        UT = U_tlist[-1,...] # take the last point which is U(t=T (input t))       

        # Now reform the Unitary to be complex (for backprop, grad calc, etc.) 
        U_Exp = UT[0,...] + 1j*UT[1,...]
        # !!! (ALERT) : This the end of the additions

        # Fidelity calulcation given by Nielsen Paper
        fidelity = 0
        d = 2**N
        for i in range(0,len(SU)):
            eps_U = torch.matmul(torch.matmul(U_Exp,SU[i]),(U_Exp.conj().T))
            target_U = torch.matmul(torch.matmul(target_gate,(SU[i].conj().T)),(target_gate.conj().T))
            tr = torch.trace(torch.matmul(target_U,eps_U))
            fidelity = fidelity + tr
        fidelity = abs(fidelity + d*d)/(d*d*(d+1))
        infidelity = 1 - fidelity
        infidelity_list[n] = infidelity.detach()
        infidelity.backward()
        
        #Printing statement
        if (n+1)%1==0: 
            print('Itertation ', str(n+1), ' out of ', str(N_iter), 'complete. Avg Infidelity: ', str(infidelity.item()))

        #optimizer 
        if optimizer is not None and scheduler is None:  # Update R
            optimizer.step()
            optimizer.zero_grad()
        elif optimizer is not None and scheduler is not None:
            optimizer.step()
            if loss_in: 
                scheduler.step(infidelity)
            else: 
                scheduler.step()
            optimizer.zero_grad()
        else:
            H1t.A.data.sub_(lr*H1t.A.grad.data) # using data avoids overwriting tensor object
            H1t.A.grad.data.zero_()           # and it's respective grad info
    
    print('The infidelity of the generated gate is: ' + str(infidelity_list.min().item()))
    return H1t.A, infidelity_list

# %% 
#Testing fidelity function
#2 qubit gates (works great :) )
target_gates =[torch.tensor((1/np.sqrt(2))*np.kron(np.array([[1,-1j],[-1j,1]]),np.eye(2)),dtype=torch.cdouble),
                torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=torch.cdouble),
                torch.tensor([[1,0,0,0],[0, 0, 1j, 0], [0, 1j,0,0],[0,0,0,1]], dtype=torch.cdouble),
                torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.cdouble),
                torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=torch.cdouble)]
#3 qubit gates (works great as well!!)
toffoli = torch.tensor([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]],dtype=torch.cdouble)
J = torch.tensor([[1,1,1],[1,1,1],[1,1,1]])
# Coef, infidelity_list = fidelity_ml(J,[1,1,1],12,toffoli,np.pi,300)

Coef, infidelity_list = fidelity_ml(J, [1,1], 6, target_gates[1],np.pi,300)

