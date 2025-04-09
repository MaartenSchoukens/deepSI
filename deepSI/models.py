import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from torch import nn
import torch
from deepSI.networks import MLP_res_net, rk4_integrator
from nonlinear_benchmarks import Input_output_data
import numpy as np
from deepSI.normalization import Norm
from warnings import warn


#######################
### Helper Function ###
#######################

def past_future_arrays(data : Input_output_data | list, na : int, nb : int, T : int | str, stride : int=1, add_sampling_time : bool=False):
    '''
    This function extracts sections from the given data as to be used in the SUBNET structure in the format (upast, ypast, ufuture, yfuture), ids. 
    
    For example for a sample [t] you will find that:
    npast = max(na,nb)
    upast[t] =   data.u[t-nb + npast : t   + npast]
    ypast[t] =   data.y[t-na + npast : t   + npast]
    ufuture[t] = data.u[t    + npast : t+T + npast]
    yfuture[t] = data.y[t    + npast : t+T + npast]

    where it can thus be used as:
       net(upast, ypast, ufuture) = y_future_sim 

    Parameters:
    - data (Input_output_data | list): Input-output data object or a list of such objects, each containing input `u` and output `y` arrays.
    - na (int): Number of past output time steps to include in the `ypast` array.
    - nb (int): Number of past input time steps to include in the `upast` array.
    - T (int or str): Length of future time window (`ufuture`, `yfuture`). If 'sim', uses the full length of the input data.
    - stride (int, optional): Step size for moving window across data (default is 1).
    - add_sampling_time (bool, optional): If True, includes a `sampling_time` array, representing sampling intervals (default is False).

    Returns:
    - Tuple of Tensors: `(upast, ypast, ufuture, yfuture, [optional sampling_time])` where each array is shaped for efficient batch training.
    - ids (np.ndarray): Indices for valid data samples, adjusted to avoid overlap when data is a list of datasets.
    '''

    # Exception handling for list of Input_output_data objects instead of a singular Input_output_data object
    if isinstance(data, (tuple,list)):
        u_past_full = y_past_full = u_future_full = y_future_full = sampling_time = torch.FloatTensor()
        ids_full = np.array([])
        past_dim = 0
        for set in data:
            (u_past_set, y_past_set, u_future_set, y_future_set), ids_set = past_future_arrays(set, na=na, nb=nb, T=T, stride=stride, add_sampling_time=False)
            u_past_full = torch.concat((u_past_full, u_past_set), dim=0)
            y_past_full = torch.concat((y_past_full, y_past_set), dim=0)
            u_future_full = torch.concat((u_future_full, u_future_set), dim=0)
            y_future_full = torch.concat((y_future_full, y_future_set), dim=0)
            ids_full = np.concat((ids_full, past_dim+ids_set)).astype(int)
            past_dim += int(set.u.shape[0])
            if add_sampling_time:
                sampling_time_set = set.sampling_time*torch.ones(u_past_set.shape[0])
                # NB: This approach takes the single provided value for the sampling time present in the data and set it as constant throughout the dataset.
                sampling_time =  torch.concat((sampling_time, sampling_time_set), dim=0)
        if add_sampling_time:
            return (u_past_full, y_past_full, u_future_full, sampling_time, y_future_full), ids_full
        else:
            return (u_past_full, y_past_full, u_future_full, y_future_full), ids_full

    # Compute number of possible windows
    n_samples = len(data.u)
    if T=="sim":    # Use full length of the input data
        n_windows = 1
        T = n_samples-max(na, nb)
    else:
        n_windows = int((n_samples-max(na, nb)-T)/stride + 1)
    
    # Check whether assumptions are met
    assert len(data.u) == len(data.y), "Input_output_data should contain input and output data of the same length."
    assert nb + T <= len(data.u), "Timewindow can not be longer than the dataset"
    assert na + T <= len(data.y), "Timewindow can not be longer than the dataset"

    # Create tensors in dimension [n_windows, time_steps] for the single input case or [n_windows, time_steps, n_states] for the multi input case
    u = torch.as_tensor(data.u)
    if len(u.shape) == 1: # Single input
        u_past = torch.as_strided(u[:-T], size=(n_windows, nb), stride=(stride, 1))
        u_future = torch.as_strided(u[nb:], size=(n_windows, T), stride=(stride, 1))
    else:   # Multiple inputs
        u_dim = u.shape[1]
        u_past = torch.as_strided(u[:-T, :], size=(n_windows, nb, u_dim), stride=(stride*u_dim, u_dim, 1))
        u_future = torch.as_strided(u[nb:, :], size=(n_windows, T, u_dim), stride=(stride*u_dim, u_dim, 1))

    # Create tensors in dimension [n_windows, time_steps] for the single output case or [n_windows, time_steps, n_states] for the multi output case
    y = torch.as_tensor(data.y)
    if len(u.shape) == 1: # Single input
        y_past = torch.as_strided(y[:-T], size=(n_windows, na), stride=(stride, 1))
        y_future = torch.as_strided(y[na:], size=(n_windows, T), stride=(stride, 1))
    else:   # Multiple inputs
        y_dim = u.shape[1]  
        y_past = torch.as_strided(y[:-T, :], size=(n_windows, na, y_dim), stride=(stride*y_dim, y_dim, 1))
        y_future = torch.as_strided(y[na:, :], size=(n_windows, T, y_dim), stride=(stride*y_dim, y_dim, 1))         
    
    # Create numpy array for the ids
    ids = np.arange(0, n_windows*stride, stride)

    # Add sampling_time tensor:
    if add_sampling_time: 
        sampling_time = data.sampling_time*torch.ones(n_windows)
        # NB: This approach takes the single provided value for the sampling time present in the data and set it as constant throughout the dataset.
        return (u_past, y_past, u_future, sampling_time, y_future), ids
    else:
        return (u_past, y_past, u_future, y_future), ids

def validate_SUBNET_structure(model):
    nx, nu, ny, na, nb = model.nx, model.nu, model.ny, model.na, model.nb
    v = lambda *size: torch.randn(size)
    xtest = v(1,nx)
    utest = v(1) if nu=='scalar' else v(1,nu)
    upast_test =  v(1, nb) if nu=='scalar' else v(1, nb, nu)
    ypast_test = v(1, na) if ny=='scalar' else v(1, na, ny)

    with torch.no_grad():
        if isinstance(model, (SUBNET, SUBNET_CT)):
            f = model.f if isinstance(model, SUBNET) else model.f_CT
            xnext_test = f(xtest, utest)
            assert xnext_test.shape==(1,nx), f'f returned the incorrect shape it should be f(x, u).shape==(nbatch=1, nx) but got {xnext_test.shape}'
            x_encoded = model.encoder(upast_test, ypast_test)
            assert x_encoded.shape==(1,nx), f'encoder returned the incorrect shape it should be model.encoder(upast, ypast).shape==(nbatch=1, nx) but got {x_encoded.shape}'
            y_pred = model.h(xtest, utest) if model.feedthrough else model.h(xtest)
            assert (y_pred.shape==(1,)) if ny=='scalar' else (y_pred.shape==(1,ny)), f'h returned the incorrect shape it should be model.h(x{", u" if model.feedthrough else ""}).shape==(nbatch=1{"" if ny=="scalar" else ", ny"}) but got {y_pred.shape}'
            if isinstance(model, SUBNET_CT):
                xnext_test = model.integrator(model.f_CT, xtest, utest, torch.ones((1,)))
                assert xnext_test.shape==(1,nx), f'integrator returned the incorrect shape it should be model.integrator(model.f_CT, x, u, Ts).shape==(nbatch=1, nx) but got {xnext_test.shape}'
        else:
            raise NotImplementedError(f'model validation of type {model} cannot be validated yet')

##############################
#### Discrete-Time SUBNET ####
##############################
# see: https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf or 
# Beintema, Gerben, Roland Toth, and Maarten Schoukens. "Nonlinear state-space identification using deep encoder networks." Learning for dynamics and control. PMLR, 2021.
# see: https://www.sciencedirect.com/science/article/pii/S0005109823003710
# Beintema, Gerben I., Maarten Schoukens, and Roland Tóth. "Deep subspace encoders for nonlinear system identification." Automatica 156 (2023): 111210.

class SUBNET(nn.Module):
    def __init__(self, nu:int|str, ny:int|str, norm : Norm, nx:int=10, nb:int=20, na:int=20, \
                 f=None, h=None, encoder=None, feedthrough=False, validate=True) -> None:
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        self.f = f if f is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = nx))
        self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        if validate:
            validate_SUBNET_structure(self)

    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride)

    def forward_simple(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        #is a lot simplier but also about 50% slower
        yfuture_sim = []
        x = self.encoder(upast, ypast)
        for u in ufuture.swapaxes(0,1):
            y = self.h(x,u) if self.feedthrough else self.h(x)
            yfuture_sim.append(y)
            x = self.f(x,u)
        return torch.stack(yfuture_sim, dim=1)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1): #unroll over time dim
            xfuture.append(x)
            x = self.f(x,u)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indicies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim (Nbatch, Ntime, ...) -> (Nbatch*Ntim, ...)
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture)) #compute the output for all time and and batches in one go
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T, ...) -> (Nbatch, T, ...)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        if data.sampling_time!=self.norm.sampling_time:
            warn('It seems that the model is being simulated at a different sampling time as it was trained on.')
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    def f_unbached(self, x, u):
        return self.f(x[None],u[None])[0]
    def h_unbached(self, x, u=None):
        return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    def encoder_unbached(self, upast, ypast):
        return self.encoder(upast[None],ypast[None])[0]

################################
#### Continuous-Time SUBNET ####
################################
# see: https://arxiv.org/abs/2204.09405
#  Beintema, G. I., Schoukens, M., & Tóth, R. (2022). Continuous-time identification  of dynamic state-space models by deep subspace encoding. Presented at the 11th International Conference on Learning Representations (ICLR)

class SUBNET_CT(nn.Module):
    #both norm, base_sampling_time have a sample time 
    def __init__(self, nu, ny, norm:Norm, nx=10, nb=20, na=20, f_CT=None, h=None, encoder=None, integrator=None, feedthrough=False, validate=True) -> None:
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.nb, self.na, self.feedthrough = nu, ny, norm, nx, nb, na, feedthrough
        self.f_CT = f_CT if f_CT is not None else norm.f_CT(MLP_res_net(input_size = [nx , nu], output_size = nx), tau=norm.sampling_time*50)
        self.h = h if h is not None else norm.h(MLP_res_net(input_size = [nx , nu] if feedthrough else nx, output_size = ny))
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        self.integrator = integrator if integrator is not None else rk4_integrator
        if validate:
            validate_SUBNET_structure(self)

    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=True)

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, sampling_time : float | torch.Tensor, yfuture: torch.Tensor=None):
        B, T = ufuture.shape[:2]
        x = self.encoder(upast, ypast)
        xfuture = []
        for u in ufuture.swapaxes(0,1):
            xfuture.append(x)
            x = self.integrator(self.f_CT, x, u, sampling_time)
        xfuture = torch.stack(xfuture,dim=1) #has shape (Nbatch, Ntime=T, nx)

        #compute output at all the future time indicies at the same time by combining the time and batch dim.
        fl = lambda ar: torch.flatten(ar, start_dim=0, end_dim=1) #conbine batch dim and time dim 
        yfuture_sim_flat = self.h(fl(xfuture), fl(ufuture)) if self.feedthrough else self.h(fl(xfuture)) #compute the output for all time and and batches in one go
        return torch.unflatten(yfuture_sim_flat, dim=0, sizes=(B,T)) #(Nbatch*T) -> (Nbatch, T)
    
    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        if data.sampling_time!=self.norm.sampling_time:
            warn('It seems that the model is being simulated at a different sampling time as it was trained on. The encoder currently assumes that the sampling_time is kept constant')
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=True)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

    def f_CT_unbached(self, x, u):
        return self.f_CT(x[None],u[None])[0]
    def integrator_unbached(self, f_CT, x, u, sampling_time):
        return self.integrator(f_CT, x[None], u[None], sampling_time[None])[0]
    def h_unbached(self, x, u=None):
        return self.h(x[None], u[None])[0] if self.feedthrough else self.h(x[None])[0]
    def encoder_unbached(self, upast, ypast):
        return self.encoder(upast[None],ypast[None])[0]

###############################################
### Helper Function for Fully Custom SUBNET ###
###############################################

class Custom_SUBNET(nn.Module):
    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=False)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=False)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

class Custom_SUBNET_CT(nn.Module):
    def create_arrays(self, data: Input_output_data | list, T : int=50, stride: int=1):
        return past_future_arrays(data, self.na, self.nb, T=T, stride=stride, add_sampling_time=True)

    def simulate(self, data: Input_output_data | list):
        if isinstance(data, (list, tuple)):
            return [self.simulate(d) for d in data]
        ysim = self(*past_future_arrays(data, self.na, self.nb, T='sim', add_sampling_time=True)[0])[0].detach().numpy()
        return Input_output_data(u=data.u, y=np.concatenate([data.y[:max(self.na, self.nb)],ysim],axis=0), state_initialization_window_length=max(self.na, self.nb))

def validate_custom_SUBNET_structure(model):
    nu, ny, na, nb = model.nu, model.ny, model.na, model.nb
    for batch_size in [1,2]:
        T = 10
        v = lambda *size: torch.randn(size)
        upast_test =  v(batch_size, nb) if nu=='scalar' else v(batch_size, nb, nu)
        ypast_test = v(batch_size, na) if ny=='scalar' else v(batch_size, na, ny)
        ufuture_test = v(batch_size, T) if nu=='scalar' else v(batch_size, T, nu)
        yfuture_test = v(batch_size, T) if ny=='scalar' else v(batch_size, T, ny)

        with torch.no_grad():
            if isinstance(model, Custom_SUBNET):
                yfuture_pred = model(upast_test, ypast_test, ufuture_test, yfuture_test)
            else:
                yfuture_pred = model(upast_test, ypast_test, ufuture_test, v(batch_size))
            assert yfuture_pred.shape==((batch_size,T) if ny=='scalar' else (batch_size,T,ny))

#########################
####### SUBNET_LPV ######
#########################
# See: https://arxiv.org/abs/2204.04060
# Verhoek, Chris, et al. "Deep-learning-based identification of LPV models for nonlinear systems." 2022 IEEE 61st Conference on Decision and Control (CDC). IEEE, 2022.

from deepSI.networks import Bilinear
class SUBNET_LPV(Custom_SUBNET):
    def __init__(self, nu, ny, norm:Norm, nx, n_schedual, na, nb, scheduling_net=None, A=None, B=None, C=None, D=None, encoder=None, feedthrough=True):
        if np.any(10*abs(norm.ymean.numpy())>norm.ystd.numpy()) or np.any(10*abs(norm.umean.numpy())>norm.ustd.numpy()):
            from warnings import warn
            warn('SUBNET_LPV assumes that the data is approximatly zero mean. Not doing so can lead to unintended behaviour.')
        assert isinstance(nu, int) and isinstance(ny, int) and isinstance(n_schedual, int) and feedthrough, 'SUBNET_LPV requires the input, output and schedualing parameter to be vectors and feedthrough to be present'
        super().__init__()
        self.nu, self.ny, self.norm, self.nx, self.n_schedual, self.na, self.nb, self.feedthrough = nu, ny, norm, nx, n_schedual, na, nb, feedthrough
        self.A = A if A is not None else Bilinear(n_in=nx, n_out=nx, n_schedual=n_schedual)
        self.B = B if B is not None else Bilinear(n_in=nu, n_out=nx, n_schedual=n_schedual, std_input=norm.ustd)
        self.C = C if C is not None else Bilinear(n_in=nx, n_out=ny, n_schedual=n_schedual, std_output=norm.ystd)
        self.D = D if D is not None else Bilinear(n_in=nu, n_out=ny, n_schedual=n_schedual, std_output=norm.ystd, std_input=norm.ustd)
        self.encoder = encoder if encoder is not None else norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx))
        self.scheduling_net = scheduling_net if scheduling_net is not None else norm.f(MLP_res_net(input_size = [nx , nu], output_size = n_schedual))
        validate_custom_SUBNET_structure(self) #does checks if forward is working as intended
    
    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        mv = lambda A, x: torch.bmm(A, x[:, :, None])[:,:,0] #batched matrix vector multiply
        yfuture_sim = []
        x = self.encoder(upast, ypast)
        for u in ufuture.swapaxes(0,1): #iterate over time
            p = self.scheduling_net(x, u)
            A, B, C, D = self.A(p), self.B(p), self.C(p), self.D(p)
            y = mv(C, x) + mv(D, u)
            x = mv(A, x) + mv(B, u)
            yfuture_sim.append(y)
        return torch.stack(yfuture_sim, dim=1)


class SUBNET_LPV_ext_scheduled(SUBNET_LPV):
    '''LPV system identification approach LPV_SUBNET with external scheduling as seen in Fig. 2 in https://arxiv.org/pdf/2204.04060'''
    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor):
        Nbatch, T = ufuture.shape[:2]
        #upasts = [upast_k, upast_k+1, u_past_k+2,...] ect
        upasts = torch.cat([upast, ufuture[:,:-1]], dim=1).unfold(1,self.nb,1).permute(0,1,3,2).flatten(start_dim=0, end_dim=1) #(Nbatch * T, nb, nu)
        ypasts = torch.cat([ypast, yfuture[:,:-1]], dim=1).unfold(1,self.na,1).permute(0,1,3,2).flatten(start_dim=0, end_dim=1) #(Nbatch * T, na, ny)

        x_long = torch.unflatten(self.encoder(upasts, ypasts), dim=0, sizes=(Nbatch, T)) #use encoder to estimate all the initial state in the future
        pfuture = torch.unflatten(self.scheduling_net(x_long.flatten(0,1), ufuture.flatten(0,1)), dim=0, sizes=(Nbatch, T)) #construct scheduling parameters
        x = x_long[:,0] #set initial state equal to the first of the inital states computed

        mv = lambda A, x: torch.bmm(A, x[:, :, None])[:,:,0] #batched matrix vector multiply
        yfuture_sim = []
        for p, u in zip(pfuture.swapaxes(0,1), ufuture.swapaxes(0,1)): #iterate over time
            A, B, C, D = self.A(p), self.B(p), self.C(p), self.D(p)
            y = mv(C, x) + mv(D, u)
            x = mv(A, x) + mv(B, u)
            yfuture_sim.append(y)
        return torch.stack(yfuture_sim, dim=1)

##########################
####### CNN_SUBNET #######
##########################
# see: https://pure.tue.nl/ws/portalfiles/portal/318935789/20240321_Beintema_hf.pdf Chapter 4
# Beintema, Gerben Izaak. PhD Thesis: "Data–driven Learning of Nonlinear Dynamic Systems: A Deep Neural State–Space Approach." (2024). Chapter 4

class CNN_SUBNET(SUBNET):
    def __init__(self, nu, ny, norm, nx, nb, na):
        from deepSI.networks import CNN_vec_to_image, CNN_encoder, MLP_res_net
        h = norm.h(CNN_vec_to_image(nx, ny=ny))
        f = norm.f(MLP_res_net(input_size=[nx, nu], output_size=nx))
        encoder = norm.encoder(CNN_encoder(nb, nu, na, ny, nx))
        super().__init__(nu, ny, norm, nx, nb, na, f, h, encoder, validate=False)

###########################
####### pHNN_SUBNET #######
###########################
# see: https://arxiv.org/abs/2305.01338
# Moradi, Sarvin, et al. "Physics-Informed Learning Using Hamiltonian Neural Networks with Output Error Noise Models." IFAC-PapersOnLine 56.2 (2023): 5152-5157.

from deepSI.networks import Ham_converter, ELU_lower_bound, Skew_sym_converter, Sym_pos_semidef_converter, Matrix_converter
class pHNN_SUBNET(Custom_SUBNET_CT):
    def __init__(self, nu : int | str, ny: int | str, norm : Norm, nx : int, na : int, nb : int, Hnet : None | nn.Module =None, Jnet : None | nn.Module =None, \
                 Rnet : None | nn.Module =None, Gnet : None | nn.Module =None, encoder : None | nn.Module =None, integrator=None, tau : float =None):
        super().__init__()
        assert nu==ny
        self.nu, self.ny, self.norm, self.nx, self.na, self.nb = nu, ny, norm, nx, na, nb
        self.Hnet = Ham_converter(ELU_lower_bound(MLP_res_net(nx, 'scalar'))) if Hnet is None else Hnet
        self.Jnet = Skew_sym_converter(MLP_res_net(nx, nx*nx)) if Jnet is None else Jnet
        self.Rnet = Sym_pos_semidef_converter(MLP_res_net(nx, nx*nx)) if Rnet is None else Rnet
        nu_val = 1 if nu=='scalar' else nu
        self.Gnet = Matrix_converter(MLP_res_net(nx, nx*nu_val), nrows=nx, ncols=nu_val) if Gnet is None else Gnet
        self.integrator = rk4_integrator if integrator is None else integrator
        self.encoder = norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)) if encoder is None else encoder
        self.norm = norm
        self.tau = norm.sampling_time*10 if tau is None else tau

        #validation of structure
        for Nbatch in [1,2]:
            xtest = torch.randn(Nbatch, nx)
            J_x, R_x, G_x, dHdx, H = self.get_matricies(xtest)
            nu_val = 1 if nu=='scalar' else nu
            assert J_x.shape == (Nbatch, nx, nx), f'Jnet(x) has the incorrect shape, expected (Nbatch={Nbatch}, nx={nx}, nx={nx}) but got Jnet(x).shape={J_x.shape}'
            assert R_x.shape == (Nbatch, nx, nx), f'Rnet(x) has the incorrect shape, expected (Nbatch={Nbatch}, nx={nx}, nx={nx}) but got Rnet(x).shape={R_x.shape}'
            assert H.shape == (Nbatch,), f'Hnet(x) has the incorrect shape, expected (Nbatch={Nbatch},) but got Hnet(x).shape={H.shape}'
            assert G_x.shape == (Nbatch, nx, nu_val), f'Gnet(x) has the incorrect shape, expected (Nbatch={Nbatch}, nx={nx}, nu_val={nu_val}) but got Gnet(x).shape={G_x.shape}'
            assert dHdx.shape == (Nbatch, nx), f'dHnet(x)/dx has the incorrect shape, expected (Nbatch={Nbatch}, nx={nx}) but got dHdx.shape={dHdx.shape}'
        validate_custom_SUBNET_structure(self)
    
    def get_matricies(self, x):
        with torch.enable_grad():
            if x.requires_grad == False:
                x.requires_grad = True
            H = self.Hnet(x)
            Hsum = H.sum()
            dHdx = torch.autograd.grad(Hsum, x, create_graph=True)[0]

        J_x = self.Jnet(x)
        R_x = self.Rnet(x)
        G_x = self.Gnet(x)
        return J_x, R_x, G_x, dHdx, H

    def forward(self, upast, ypast, ufuture, sampling_time, yfuture=None):
        x = self.encoder(upast, ypast)
        ufuture = (ufuture.view(ufuture.shape[0],ufuture.shape[1],-1)-self.norm.umean)/self.norm.ustd #normalize inputs
        yfuture_sim = []
        for u in ufuture.swapaxes(0,1): #if using a 1-step euler this can be reduced further 
            J_x, R_x, G_x, dHdx, H = self.get_matricies(x) #this can be done outside of the loop for a speedup
            y_hat = torch.einsum('bij,bi->bj', G_x, dHdx) #bij,bi->bj  = A^T @ dHdx
            yfuture_sim.append(y_hat)
            def f_CT(x, u):
                J_x, R_x, G_x, dHdx, H = self.get_matricies(x)
                Gu = torch.einsum('bij,bj->bi', G_x, u) # G_x (Nb, nx, nu) times u (Nb, nu) = (Nb, nx)
                return (torch.einsum('bij,bj->bi', J_x - R_x, dHdx) + Gu)/self.tau
    
            x = self.integrator(f_CT, x, u, sampling_time)
        
        yfuture_sim = torch.stack(yfuture_sim, dim=1)
        yfuture_sim = yfuture_sim[:,:,0] if self.ny=='scalar' else yfuture_sim
        return yfuture_sim*self.norm.ystd + self.norm.ymean

##############################
####### Koopman SUBNET #######
##############################
# see: https://ieeexplore.ieee.org/abstract/document/9682946
# Iacob, Lucian Cristian, et al. "Deep identification of nonlinear systems in Koopman form." 2021 60th IEEE Conference on Decision and Control (CDC). IEEE, 2021.

class Koopman_SUBNET(Custom_SUBNET):
    '''Implements the following structure
    x_next = A@x + B(x)@(u - umean)/ustd
    y = (C@x) * ystd + ymean

    if feedthrough: y = (C@x) * ystd + ymean + (D@(u - umean)/ustd) * ystd + ymean 
    if B_depends_on_u: x_next = A@x + B(x, u)@(u - umean)/ustd
        
    '''
    def __init__(self, nu, ny, norm : Norm, nx, nb, na, encoder=None, A=None, Bnet=None, C=None, D=None, B_depends_on_u=False, feedthrough=False):
        super().__init__()
        from deepSI.networks import Matrix_converter, MLP_res_net
        self.nu, self.ny, self.norm, self.nx, self.na, self.nb, self.feedthrough, self.B_depends_on_u = nu, ny, norm, nx, na, nb, feedthrough, B_depends_on_u
        self.nu_vals = 1 if nu=='scalar' else nu
        self.ny_vals = 1 if ny=='scalar' else ny
        self.encoder = norm.encoder(MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx)) if encoder is None else encoder

        self.A = nn.Parameter(torch.randn((nx,nx))/(2*nx**0.5)) if A==None else A
        self.Bnet = Matrix_converter(MLP_res_net([nx, nu] if B_depends_on_u else nx, nx*self.nu_vals), nrows=nx, ncols=self.nu_vals) if Bnet==None else Bnet
        self.C = nn.Parameter(torch.randn((self.ny_vals,nx))/(2*nx**0.5)) if C==None else C
        if feedthrough:
            self.D = nn.Parameter(torch.randn((self.ny_vals,self.nu_vals))/(2*self.nu_vals**0.5)) if D==None else D
        else:
            self.D = None

    def forward(self, upast: torch.Tensor, ypast: torch.Tensor, ufuture: torch.Tensor, yfuture: torch.Tensor=None):
        mv = lambda A, x: torch.bmm(A, x[:, :, None])[:,:,0] #batched matrix vector multiply
        yfuture_sim = []
        x = self.encoder(upast, ypast) #initial state
        Nbatch = upast.shape[0]
        ufuture = (ufuture - self.norm.umean)/self.norm.ustd # Normalize input
        ufuture = ufuture.view(Nbatch, ufuture.shape[1], -1) # Convert all the u from scalars to vectors if needed
        # Add batch dimension to matrixes
        A = torch.broadcast_to(self.A, (Nbatch, self.nx, self.nx)) 
        C = torch.broadcast_to(self.C, (Nbatch, self.ny_vals, self.nx))
        D = None if self.feedthrough==False else torch.broadcast_to(self.D, (Nbatch, self.ny_vals, self.nu_vals))
        for u in ufuture.swapaxes(0,1): #iterate over time
            y = mv(C, x) + (0 if self.feedthrough==False else mv(D, u))
            yfuture_sim.append(y)
            B = self.Bnet(x) if self.B_depends_on_u==False else self.Bnet(x, u[:,0] if self.nu=='scalar' else u) #removes the vector dim if it is scalar
            x = mv(A,x) + mv(B, u)
        yfuture_sim = torch.stack(yfuture_sim, dim=1)
        if self.ny=='scalar':
            yfuture_sim = yfuture_sim[:,:,0]
        return yfuture_sim*self.norm.ystd + self.norm.ymean
