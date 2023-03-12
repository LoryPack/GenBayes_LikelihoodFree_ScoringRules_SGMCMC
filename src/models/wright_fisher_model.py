import torch
import torchsde
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torchsde import BrownianInterval
from einops import rearrange


class WFSDE():
    def __init__(self, name:str='WFSDE', seed:int=42, simulation_method='euler',y0=None, snp_size = 100,population_size = 1000,generation=60,interval=1, priors:List=None):
        """
        

        Parameters
        ----------
        simulation_method : Str
            Specify method for solving SDE. The default is 'euler'.
        y0 : Array, optional
            Initial allele frequencies. The default is None, will be 0.2 for all allele.
        snp_size : int
            Number of SNPs. The default is 100.
        population_size : int
            Population Size. The default is 1000.
        generation : int, 
            Specify number of generations. The default is 60.
        interval : int, 
            Specify generation interval for producing allele frequencies. The default is 1.
            1 means produce output for every generation.
        priors : List, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        1d array by solving Wright-Fisher diffusion SDE.

        """
        self.seed = seed
        self.param_dim = 2 #check this
        self.simulation_method = simulation_method
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None
        self.snp_size = snp_size
        self.population_size = population_size
        self.y0 = y0
        self.generation = generation
        self.interval = interval
        
        if self.y0 == None:
            # user can specify initial allele frequencies or by default set to 0.2
            self.y0 = torch.full((1, self.snp_size),0.2)
        else:
            if self.y0.shape[1] != self.snp_size:
               raise TypeError('Length of initial allele frequencies should match the number of SNPs')
            if torch.sum((self.y0>0) & (self.y0<1)) != self.snp_size:
               raise TypeError('Initial allele frequencies should between 0 and 1')
            
        self.dt = 5e-5
        self.ts = torch.linspace(0,self.generation,int(self.generation/self.interval) + 1)/(2*self.population_size)


    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int=None):
        """
        params: List of two parameters 
        num_forward_simulations: int
        """
        if seed is None:
            torch.manual_seed(self.seed)
            self.seed += 1
        else:
            torch.manual_seed(seed)

        batch_size = num_forward_simulations

        self.model = torch_WFSDE(self.population_size,self.snp_size,*params)
        bm = BrownianInterval(t0=self.ts[0], 
                      t1=self.ts[-1], 
                      size=(batch_size, self.snp_size), entropy=self.seed)
        ys = torchsde.sdeint(self.model, self.y0.repeat(batch_size, 1), self.ts, dt=self.dt, method=self.simulation_method, bm=bm)  # ys will have shape (t_size, batch_size, state_size)
        ys = rearrange(ys, 't b d -> b (t d)')  # in this way return 1d arrays as outputs

        return ys

    def get_output_dimension(self):
        return self.snp_size #Check this

    def _check_input(self, input_values):
        return True

    def _check_output(self, values):
        return True


class WFSDE_ha():
    def __init__(self, name:str='WFSDE_ha', seed:int=42, simulation_method='euler', y0=None,population_size = 1000,generation=60,interval=1, priors:List=None):
        """
        Model for 4 Haplotypes

        Parameters
        ----------
        name : str, optional
            DESCRIPTION. The default is 'WFSDE_ha'.
        seed : int, optional
            DESCRIPTION. The default is 42.
        simulation_method : str
            Specify method for solving SDE. The default is 'euler'.
        y0 : array, optional
            Initial haplotype frequencies. The default is None.
        population_size : int,
            Population size. The default is 1000.
        generation : int, 
            Specify number of generations. The default is 60.
        interval : int, 
            Specify generation interval for producing haplotype frequencies. The default is 1.
            1 means produce output for every generation.
        priors : List, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        1d array for 4 haplotype frequencies over specified generations.

        """
        self.seed = seed
        self.param_dim = 2 #check this
        self.simulation_method = simulation_method
        self.scores = [] # for debugging purposes, just to save the score
        self.model = None
        self.population_size = population_size
        self.generation = generation
        self.interval = interval
        self.y0  = y0
        if self.y0 == None:
            self.y0 = torch.full((1,4),0.25)
        else:
            if self.y0.shape[1] != 4:
                raise TypeError('Number of haplotype have to be 4, the shape of initial haplotype frequency should be (1,4)')
            if torch.sum((self.y0>0) & (self.y0<1)) != 4:
                raise TypeError('Initial haplotype frequencies should between 0 and 1')
            if torch.sum(self.y0) != 1:
                raise TypeError('Initial haplotype frequencies have to sum up to 1')
            
            
        self.dt = 5e-5
        
        self.ts = torch.linspace(0,self.generation,int(self.generation/self.interval) + 1)/(2*self.population_size)
   

    def torch_forward_simulate(self, params, num_forward_simulations: int, seed:int=None):
        #TODO: Need to fix batch sizes in general noise
        pass

    def get_output_dimension(self):
        return self.snp_size #Check this

    def _check_input(self, input_values):
        return True

    def _check_output(self, values):
        return True

class torch_WFSDE(torch.nn.Module):
    # diagonal noise
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self,N,snp_n,s,h):
        """
        SDE for diploid Wright-Fisher Diffusion

        Parameters
        ----------
        N : int
            Population size.
        s : array
            Selection coefficients for each locus.
        h : numeric between 0 and 1
            dominance coefficient.

        Returns
        -------
        None.

        """
        super().__init__()
        self.N = N
        self.s = s
        self.h = h
        self.snp_n = snp_n

        N = torch.Tensor([self.N]).repeat(self.snp_n)
        h = torch.Tensor([self.h]).repeat(self.snp_n)
        # rescale selection coefficient
        self.Ns = 2*N*self.s
        

    # Drift
    def f(self,t,x):
        return self.Ns*x*(1-x)*((1-self.h)-(1-2*self.h)*x)

    # Diffusion
    def g(self,t,x):
        return torch.sqrt(x*(1-x))

class torch_WFSDE_ha(torch.nn.Module):
    # diagonal noise
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self,N,haplo_n,s,r,h):
        """
        SDE for diploid two linked loci Wright-Fisher Diffusion

        Parameters
        ----------
        N : int
            Population size.
        s : array
            Selection coefficients for each locus.
        r: numeric
            recombination rate between two loci
        h : array
            dominance coefficient for each locus.
        

        Returns
        -------
        SDE for two linked loci wright fisher diffusion.

        """
        super().__init__()
        self.N = N
        self.s = s
        self.haplo_n = haplo_n
        self.h = h
        self.r = r

        # rescale selection coefficient and recombination rate
        self.Nsa = 2*self.N*self.s[0]
        self.Nsb = 2*self.N*self.s[1]
        self.rho = 4*self.N*self.r

        

    # Drift
    def f(self,t,x):
        mu = torch.zeros(1,4)
        mu[0,0] = self.Nsa*x[0,0]*(x[0,2]+x[0,3])*((x[0,0]+x[0,1])*self.h[0] + (x[0,2]+x[0,3])*(1-self.h[0]))+ \
                    self.Nsb*x[0,0]*(x[0,1]+x[0,3])*((x[0,0]+x[0,2])*self.h[1] + (x[0,1]+x[0,3])*(1-self.h[1]))- \
                        (self.rho/2)*(x[0,0]*x[0,3] - x[0,1]*x[0,2])
                    
        mu[0,1] = self.Nsa*x[0,1]*(x[0,2]+x[0,3])*((x[0,0]+x[0,1])*self.h[0] + (x[0,2]+x[0,3])*(1-self.h[0]))- \
                    self.Nsb*x[0,1]*(x[0,0]+x[0,2])*((x[0,0]+x[0,2])*self.h[1] + (x[0,1]+x[0,3])*(1-self.h[1]))+ \
                        (self.rho/2)*(x[0,0]*x[0,3] - x[0,1]*x[0,2])
                        
        mu[0,2] = -self.Nsa*x[0,2]*(x[0,0]+x[0,1])*((x[0,0]+x[0,1])*self.h[0] + (x[0,2]+x[0,3])*(1-self.h[0]))+ \
                    self.Nsb*x[0,2]*(x[0,1]+x[0,3])*((x[0,0]+x[0,2])*self.h[1] + (x[0,1]+x[0,3])*(1-self.h[1]))+ \
                        (self.rho/2)*(x[0,0]*x[0,3] - x[0,1]*x[0,2])
                        
        mu[0,3] = -self.Nsa*x[0,3]*(x[0,0]+x[0,1])*((x[0,0]+x[0,1])*self.h[0] + (x[0,2]+x[0,3])*(1-self.h[0]))- \
                    self.Nsb*x[0,3]*(x[0,0]+x[0,2])*((x[0,0]+x[0,2])*self.h[1] + (x[0,1]+x[0,3])*(1-self.h[1]))- \
                        (self.rho/2)*(x[0,0]*x[0,3] - x[0,1]*x[0,2])
        return mu

    # Diffusion
    def g(self,t,x):
        sigma = torch.zeros(4,6)
        sigma[0,0] = torch.sqrt(x[0,0]*x[0,1])
        sigma[0,1] = torch.sqrt(x[0,0]*x[0,2])
        sigma[0,2] = torch.sqrt(x[0,0]*x[0,3])
        
        sigma[1,0] = -torch.sqrt(x[0,0]*x[0,1])
        sigma[1,3] = torch.sqrt(x[0,2]*x[0,1])
        sigma[1,4] = torch.sqrt(x[0,3]*x[0,1])
        
        sigma[2,1] = -torch.sqrt(x[0,0]*x[0,2])
        sigma[2,3] = -torch.sqrt(x[0,2]*x[0,1])
        sigma[2,5] = torch.sqrt(x[0,2]*x[0,3])
        
        sigma[3,2] = -torch.sqrt(x[0,3]*x[0,0])
        sigma[3,4] = -torch.sqrt(x[0,3]*x[0,1])
        sigma[3,5] = -torch.sqrt(x[0,3]*x[0,2])
        
                    
        return torch.reshape(sigma,(1,4,6))
