
from deepSI.systems.System import System, System_IO, System_data, load_system
import numpy as np
from deepSI.system_data.datasets import get_work_dirs
import deepSI
import torch
from torch import nn, optim
from tqdm.auto import tqdm
import time

class System_fittable(System):
    """docstring for System_fit"""
    def fit(self,sys_data,**kwargs):
        if self.fitted==False:
            self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
        self._fit(self.norm.transform(sys_data),**kwargs) #transfrom data to fittable data?
        self.fitted = True

class System_IO_fit_sklearn(System_fittable, System_IO): #name?
    def __init__(self, na, nb, reg):
        super(System_IO_fit_sklearn, self).__init__(na, nb)
        self.reg = reg

    def _fit(self,sys_data):
        #sys_data #is already normed fitted on 
        hist,y = sys_data.to_IO_data(na=self.na,nb=self.nb)
        self.reg.fit(hist,y)

    def IO_step(self,uy):
        return self.reg.predict([uy])[0] if uy.ndim==1 else self.reg.predict(uy)


class System_PyTorch(System_fittable):
    """docstring for System_PyTorch"""
    # def __init__(self, arg):
    #     super(System_PyTorch, self).__init__()
    #     self.arg = arg

    def init_nets(self,nu,ny):
        #returns parameters
        raise NotImplementedError

    def init_optimizer(self,parameters,**optimizer_kwargs):
        #return the optimizer with a optimizer.zero_grad and optimizer.step method
        if optimizer_kwargs.get('optimizer') is not None:
            optimizer = optimizer_kwargs['optimizer']
            del optimizer_kwargs['optimizer']
        else:
            optimizer = torch.optim.Adam
        return optimizer(parameters,**optimizer_kwargs) 

    def make_training_data(self,sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        raise NotImplementedError


    def fit(self, sys_data, epochs=30, batch_size=256, Loss_kwargs={}, optimizer_kwargs={}, sim_val=None, verbose=1, val_frac = 0.2):
        #todo implement verbose

        #1. init funcs already happened
        #2. init optimizer
        #3. training data
        #4. optimization

        def validation():
            global time_val
            t_start_val = time.time()
            if sim_val is not None:
                sim_val_data = self.apply_experiment(sim_val)
                Loss_val = sim_val_data.NRMS(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.CallLoss(*data_val,**Loss_kwargs).item()
            time_val += time.time() - t_start_val
            self.Loss_val.append(Loss_val)
            if self.bestfit>Loss_val:
                if verbose: print('########## new best ###########')
                self.checkpoint_save_system()
                self.bestfit = Loss_val
            return Loss_val

        if self.fitted==False:
            self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
            self.paremters = list(self.init_nets(self.nu,self.ny))
            self.optimizer = self.init_optimizer(self.paremters,**optimizer_kwargs)
            self.bestfit = float('inf')
            self.Loss_val,self.Loss_train,self.batch_id,self.time = [],[],[],[]
            self.batch_counter = 0
            extra_t = 0
            self.fitted = True
        else:
            self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
            extra_t = 0 if len(self.time)==0 else self.time[-1]


        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data, **Loss_kwargs)
        data_full = [torch.tensor(dat, dtype=torch.float32) for dat in data_full]


        if sim_val is not None:
            data_train = data_full
        else:
            split = int(len(data_full)*(1-val_frac))
            data_train = [dat[:split] for dat in data_full]
            data_val = [dat[split:] for dat in data_full]

        
        global time_val
        time_val = time_back = time_loss = 0
        Loss_val = validation()
        time_val = 0 #reset
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        print(f'N_training_samples={N_training_samples}, batch_size={batch_size}, N_batch_updates_per_epoch={N_batch_updates_per_epoch}')
        ids = np.arange(0, N_training_samples, dtype=int)
        try:
            self.start_t = time.time()
            for epoch in tqdm(range(epochs)):
                np.random.shuffle(ids)

                Loss_acc = 0
                for i in range(batch_size, N_training_samples + 1, batch_size):
                    ids_batch = ids[i-batch_size:i]
                    train_batch = [part[ids_batch] for part in data_train]
                    start_t_loss = time.time()
                    Loss = self.CallLoss(*train_batch, **Loss_kwargs)
                    time_loss += time.time() - start_t_loss

                    self.optimizer.zero_grad()

                    start_t_back = time.time()
                    Loss.backward()
                    time_back += time.time() - start_t_back

                    self.optimizer.step()
                    Loss_acc += Loss.item()
                Loss_acc /= N_batch_updates_per_epoch
                self.batch_counter += N_batch_updates_per_epoch
                self.batch_id.append(self.batch_counter)
                self.Loss_train.append(Loss_acc)
                self.time.append(time.time()-self.start_t+extra_t)

                Loss_val = validation()
                if verbose>0: 
                    time_elapsed = time.time()-self.start_t
                    print(f'Epoch: {epoch+1:4} Training loss: {self.Loss_train[-1]:7.4} Validation loss = {Loss_val:6.4}, time Loss: {time_loss/time_elapsed:.1%}, back: {time_back/time_elapsed:.1%}, val: {time_val/time_elapsed:.1%}')
                # print(f'epoch={epoch} NRMS={Loss_val:9.4%} Loss={Loss_acc:.5f}')
        except KeyboardInterrupt:
            print('stopping early due to KeyboardInterrupt')
        self.checkpoint_load_system()

    def CallLoss(*args,**kwargs):
        #kwargs are the settings
        #args is the data
        raise NotImplementedError
    
    ########## Saving and loading ############
    def checkpoint_save_system(self):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints']
        self._save_system_torch(file=os.path.join(directory,self.name+'_best'+'.pth')) #error here if you have 
        vars = self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time
        np.savez(os.path.join(directory,self.name+'_best'+'.npz'),*vars)
    def checkpoint_load_system(self):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints'] 
        self._load_system_torch(file=os.path.join(directory,self.name+'_best'+'.pth'))
        out = np.load(os.path.join(directory,self.name+'_best'+'.npz'))
        out_real = [(a[1].tolist() if a[1].ndim==0 else a[1]) for a in out.items()]
        self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time = out_real
        self.Loss_val, self.Loss_train, self.batch_id, self.time = self.Loss_val.tolist(), self.Loss_train.tolist(), self.batch_id.tolist(), self.time.tolist()
        
    def _save_system_torch(self,file):
        save_dict = {}
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,(nn.Module,optim.Optimizer)):
                save_dict[d] = attribute.state_dict()
        torch.save(save_dict,file)
    def _load_system_torch(self,file):
        save_dict = torch.load(file)
        for key in save_dict:
            attribute = self.__getattribute__(key)
            try:
                attribute.load_state_dict(save_dict[key])
            except (AttributeError, ValueError):
                print('Error loading key',key)


import torch
from torch import nn

class System_Torch_IO(System_PyTorch, System_IO):
    def __init__(self,na,nb):
        super(System_Torch_IO, self).__init__(na,nb)

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        return sys_data.to_IO_data(na=self.na,nb=self.nb) #np.array(hist), np.array(Y)

    def init_nets(self, nu, ny):
        assert ny==None
        #returns parameters
        nu = 1 if nu is None else nu
        one_out = ny==None
        ny = 1 if ny is None else ny
        n_in = nu*self.nb + ny*self.na
        IN = [nn.Linear(n_in,64),nn.Tanh(),nn.Linear(64,ny),nn.Flatten()]
        self.net = nn.Sequential(*IN)
        return self.net.parameters()

    def CallLoss(self,hist,Y, **kwargs):
        return torch.mean((self.net(hist)[:,0]-Y)**2)

    def IO_step(self,uy):
        uy = torch.tensor(uy,dtype=torch.float32)
        if uy.ndim==1:
            uy = uy[None,:]
            return self.net(uy)[0,0].item()
        else:
            return self.net(uy)[:,0].detach().numpy()



def fit_system_tuner(fit_system, sys_data, search_dict, verbose=1):
    import copy
    #example use: print(hyper_parameter_tunner(System_IO_fit_linear,dict(na=[1,2,3],nb=[1,2,3]),sys_data))
    def itter(itter_dict, k=0, dict_now=None, best_score=float('inf'), best_sys=None, best_dict=None):
        if dict_now is None:
            dict_now = dict()
        if k==len(itter_dict):
            sys = fit_system(**dict_now)
            sys.fit(sys_data)
            try:
                score = sys.apply_experiment(sys_data).NRMS(sys_data)
            except ValueError:
                score = float('inf')
            if verbose>0: print(score, dict_now)
            if score<=best_score:
                return score, sys, copy.deepcopy(dict_now)
            else:
                return best_score, best_sys, best_dict
        else:
            for item in itter_dict[k][1]:
                dict_now[itter_dict[k][0]] = item
                best_score, best_sys, best_dict = itter(itter_dict, k=k+1, dict_now=dict_now, best_score=best_score, best_sys=best_sys, best_dict=best_dict)
            return best_score, best_sys, best_dict

    itter_dict = [list(a) for a in search_dict.items()]
    for k in range(len(itter_dict)):
        I = itter_dict[k][1]
        if isinstance(I,range):
            itter_dict[k][1] = list(I)
        elif not isinstance(I,(tuple,list)):
            itter_dict[k][1] = list([I])
    best_score, best_sys, best_dict = itter(itter_dict)
    if verbose>0: print('Result:', best_score, best_sys, best_dict)
    return best_sys, best_score, best_dict

if __name__ == '__main__':
    from sklearn import linear_model 
    from matplotlib import pyplot as plt
    class System_IO_fit_linear(System_IO_fit_sklearn):
        def __init__(self,na,nb):
            super(System_IO_fit_linear,self).__init__(na,nb,linear_model.LinearRegression())

    train, test = deepSI.datasets.Cascaded_Tanks()
    sys = System_Torch_IO(na=5,nb=5)
    # sys = System_encoder(nx=8, na=50, nb=50)
    # sys0 = deepSI.systems.sys_ss_test()
    # sys_data = sys0.apply_experiment(System_data(u=np.random.normal(size=10000)))

    # sys = System_IO_fit_linear(7,3)
    # sys = System_encoder(nx=8,na=20,nb=20)
    # sys_data = System_data(u=np.random.normal(size=100),y=np.random.normal(size=100))
    sys.fit(train,epochs=1000,Loss_kwargs=dict(nf=15),batch_size=8,sim_val=None,optimizer_kwargs=dict(optimizer=optim.Adam,lr=1e-3))
    print(sys.optimizer)
    # sys.save_system('../../testing/test-fit.p')
    # del sys
    # sys = load_system('../../testing/test-fit.p')

    sys_data_predict = sys.apply_experiment(test)
    print(sys_data_predict.NRMS(test))
    plt.plot(sys.n_step_error(test))
    plt.show()

    # sys_data_predict2 = sys.apply_experiment(sys_data)

    test.plot()
    sys_data_predict.plot(show=True)
    # plt.plot(test.u)
    # plt.show()
    # sys_data_predict2.plot(show=True)

