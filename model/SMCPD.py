import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial
import pandas as pd
import time

###########################################################################################################################################
     
### Bayesian Hawkes process model    
    
class HAWKES_SMC:
    def __init__(self, tt, T0):
        self.T0 = T0
        self.tt = np.array(tt)
        self.m = self.tt.size
        if self.m > 1:            
            self.dtt = np.tril( self.tt[1:, np.newaxis] - self.tt[np.newaxis, :self.m-1] )
                  
        self.DoF = 3
        self.mu_idx = 0
        self.gamma_idx = 1
        self.delta_idx = 2
        
        self.hyperMean = np.zeros( (self.DoF, 1) )
        self.hyperVar  = np.ones( (self.DoF, 1) )
    
    ## Posterior inference 
    
    def getMinusLogPrior(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        
        shift = thetas - self.hyperMean
        tmp = 0.5 * np.sum( shift ** 2 / self.hyperVar, 0 )
        return tmp if nSamples > 1 else tmp.squeeze()
        
    def getIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m > 1:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            tmp = np.vstack( ( mus, \
              mus + gammas * ( \
                np.sum( np.exp( - deltas * self.dtt[:,:,np.newaxis] ), 1 ) \
                - np.arange(self.m - 2,-1,-1)[:,np.newaxis] \
                             ) \
                             ) )
        elif self.m == 1:
            tmp = mus
        tmp = np.maximum(tmp, 1e-6)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getCompensator(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m > 1:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            tmp = mus * (self.tt[-1] - self.T0) + gammas / deltas * ( \
                    self.m - 1 - np.sum( np.exp( - deltas * self.dtt[-1,:,np.newaxis] ), 0 ) \
                                                        )
        elif self.m == 1:
            tmp =  mus * (self.tt - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getMinusLogLikelihood(self, thetas):
        if self.m > 1:
            return self.getCompensator(thetas) - np.sum( np.log( self.getIntensity(thetas) ), 0 )
        else:
            return self.getCompensator(thetas) - np.log( self.getIntensity(thetas) )
    
    def getMinusLogPosterior(self, thetas):
        if self.m == 0:
            return self.getMinusLogPrior(thetas)
        else:
            return self.getMinusLogPrior(thetas) + self.getMinusLogLikelihood(thetas)
    
    def getGradientMinusLogPrior(self, thetas):        
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        tmp = (thetas - self.hyperMean) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientLogIntensity(self, thetas):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        gammas = expthetas[self.gamma_idx,:]
        deltas = expthetas[self.delta_idx,:]
        if (nSamples > 1) or (self.m == 1):
            lams = self.getIntensity(thetas)
        else:
            lams = np.array(self.getIntensity(thetas))[:,np.newaxis]
    
        if self.m > 1:
            expmdeltasdtt = np.exp( - deltas * self.dtt[:,:,np.newaxis] )
            sumexpmdeltasdtt = np.sum( expmdeltasdtt, 1 ) \
                                    - np.arange(self.m - 2,-1,-1)[:,np.newaxis]
            dttexpmdeltasdtt = np.sum( self.dtt[:,:,np.newaxis] * expmdeltasdtt, 1 )

            gllams = np.zeros( (self.DoF, self.m, nSamples ) )
            gllams[self.mu_idx,:,:]    = mus / lams
            gllams[self.gamma_idx,:,:] = gammas * np.vstack( \
                                (np.zeros(nSamples), sumexpmdeltasdtt / lams[1:]) )
            gllams[self.delta_idx,:,:] = deltas * np.vstack( \
                                (np.zeros(nSamples), - gammas * dttexpmdeltasdtt / lams[1:]) )
            return gllams
        elif self.m == 1:
            return np.vstack( (np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
               
    def getGradientMinusLogLikelihood(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        thetas = thetas.reshape(self.DoF, nSamples)
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 )  
        mus = expthetas[self.mu_idx,:]
        gammas = expthetas[self.gamma_idx,:]
        deltas = expthetas[self.delta_idx,:]
        
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        # Expressions    
        if self.m > 1:
            expmdeltasdtm_tt = np.exp( - deltas * self.dtt[-1,:,np.newaxis] )
            f = self.m - 1 - np.sum( expmdeltasdtm_tt, 0 )
            df = np.sum( self.dtt[-1,:,np.newaxis] * expmdeltasdtm_tt, 0 )

            gcomp = np.zeros( (self.DoF, nSamples) )
            gcomp[self.mu_idx,:]    = mus * (self.tt[-1] - self.T0)
            gcomp[self.gamma_idx,:] = f * gammas / deltas
            gcomp[self.delta_idx,:] = gammas * ( df - f / deltas )
            tmp = gcomp - np.sum(gllams, 1)
        elif self.m == 1:
            gcomp = np.vstack( \
                        (mus * (self.tt - self.T0) * np.ones(nSamples), np.zeros( (self.DoF-1, nSamples) ) ) )
            tmp = gcomp - gllams
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getGradientMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = self.getGradientMinusLogPrior(thetas) \
            + self.getGradientMinusLogLikelihood(thetas, gllams)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getAsymptHessianMinusLogPosterior(self, thetas, *arg):
        nSamples = thetas.size // self.DoF
        if len(arg) == 0:
            gllams = self.getGradientLogIntensity(thetas)
        else:
            gllams = arg[0]
        tmp = np.sum( \
            gllams.reshape(self.DoF,1,self.m,nSamples) * \
            gllams.reshape(1,self.DoF,self.m,nSamples), 2) \
                + np.eye(self.DoF).reshape(self.DoF, self.DoF, 1) / self.hyperVar
        return tmp if nSamples > 1 else tmp.squeeze()       
                
    ## Prediction inference
    
    def getPredIntensity(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m != 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]
            
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = t - self.tt
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]

            tmp = mus + gammas * np.sum( np.exp( - deltas * self.dt_tt ), 0 )
        else:
            tmp = mus
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredCompensator(self, thetas, t, *arg):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        if self.m != 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]
            
            if len(arg) < 2:
                if self.m == 1:
                    self.dtm_tt = 0
                else:
                    self.dtm_tt = np.hstack( (0, self.dtt[-1,:]) )[:,np.newaxis]
            else:
                self.dtm_tt = arg[1]
            if len(arg) < 1:
                if self.m == 1:
                    self.dt_tt = np.array([t - self.tt])
                else:
                    self.dt_tt = (t - self.tt)[:,np.newaxis]
            else:
                self.dt_tt = arg[0]

            tmp =  mus * self.dt_tt[-1] + gammas / deltas * \
                np.sum( np.exp( - deltas * self.dtm_tt ) \
                       - np.exp( - deltas * self.dt_tt ) , 0 )
        else:
            tmp =  mus * (t - self.T0)
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getPredMinusLogLikelihood(self, thetas, t):
        return self.getPredCompensator(thetas, t) - np.log( self.getPredIntensity(thetas, t) )
    
    def simulateNewEvent(self, thetas):
        nSamples = thetas.size // self.DoF
        expthetas = np.maximum( np.exp( np.minimum( thetas.reshape(self.DoF, nSamples), 700 ) ), 1e-8 ) 
        mus = expthetas[self.mu_idx,:]
        S2 = - np.log( np.random.uniform(size = nSamples) ) / mus
        if self.m > 0:
            gammas = expthetas[self.gamma_idx,:]
            deltas = expthetas[self.delta_idx,:]

            lamsplus_mus = gammas * \
                (1 + np.sum( np.exp( - deltas * self.dtt[-1,:,np.newaxis] ), 0 ) ) \
                if self.m != 1 else gammas
            lamsplus_mus = np.maximum( lamsplus_mus, 1e-8 )

            D = 1 + deltas * np.log( np.random.uniform(size = nSamples) ) / lamsplus_mus
            S = np.zeros(nSamples)
            idxplus = np.where(D > 0)
            idxminus = np.where(D <= 0)
            S[idxplus] = np.min( np.vstack( (- np.log( D[idxplus] ) \
                                / deltas[idxplus], S2[idxplus]) ), 0 )
            S[idxminus] = S2[idxminus]
            tmp = S
        else:
            tmp = S2       
        return tmp if nSamples > 1 else tmp.squeeze()
    
    def getMAP(self, *arg):
        x0 = np.random.normal(size = self.DoF) if len(arg) == 0 else arg[0]
        res = optimize.minimize(self.getMinusLogPosterior, x0, method='L-BFGS-B')
        return res.x
               
###########################################################################################################################################    
### Sequential Monte Carlo (SMC) with adaptive systematic resampling. 
##  Importance density: Laplace approximation

class SMC:
    def __init__(self, model):
        self.model = model
        self.DoF   = model.DoF         
        self.nParticles = 25000            
                        
    def apply(self):
        self.importanceStatistics()
        self.particles = self.getImportanceSamples()
        self.weights   = self.getImportanceWeights()
        if self.isResamplingRequired() == True:
            self.resampleParticles()
        
    def importanceStatistics(self):
        self.MAP = self.model.getMAP().reshape(self.DoF, 1)
        gllams = self.model.getGradientLogIntensity(self.MAP)
        self.H   = self.model.getAsymptHessianMinusLogPosterior(self.MAP, gllams)
        self.C   = np.linalg.inv(self.H)
            
    def getImportanceSamples(self):
        return np.random.multivariate_normal(np.ndarray.flatten(self.MAP), self.C, self.nParticles).T
        
    def getImportanceWeights(self):
        alpha = np.exp( - self.model.getMinusLogPosterior(self.particles) \
                        + self.getMinusLogImportanceDensity(self.particles) ) 
        return alpha / np.sum(alpha)
        
    def resampleParticles(self):
        U = ( np.random.uniform() + np.arange(self.nParticles) ) / self.nParticles
        cumWeights = np.cumsum(self.weights)
        N_i = np.zeros(self.nParticles, 'i')
        i, j = 0, 0
        while i < self.nParticles:
            if U[i] < cumWeights[j]:
                N_i[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[:,N_i]
        self.weights   = np.ones(self.nParticles) / self.nParticles
        
    def getMinusLogImportanceDensity(self, thetas):
        shift = thetas - self.MAP
        return 0.5 * np.sum(shift * np.matmul(self.H, shift), 0) 
           
    def isResamplingRequired(self):
        return 2 < self.nParticles * np.sum(self.weights ** 2) 
                   
###########################################################################################################################################  
### Generalized BOCPD via SMC

class SMCPD:
    def __init__(self, data):
        self.predictprob = []
        # Import data
        self.data = data
        self.nData = len(self.data)
        
        # Setup computational effort
        self.rmax = 30      # Max run length 
        self.npts = 100    # Number of posterior samples
        self.nrls = 100     # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail  -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l = 8.5           # Left percentage of probability risk     
        self.risk_level_r = 1.5           # Right percentage of probability risk
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)
                    
        # Changepoint prior
        self.rlr = 1000             # Run length rate hyper-parameter  
        self.H   = 1 / self.rlr    # Hazard rate
        
        # Initialize run length probabilities
        self.jp  = 1          # Joint 
        self.rlp = self.jp    # Posterior
        self.t0 = [0]
        
        # Initialize models
        self.model = {}
        self.model[0] = HAWKES_SMC([], 0)
        
        # Initialize SMC
        self.smc   = {}        
        
        # Initialize posterior samples and probabilities
        self.pts = {}             
        self.pts[0] = np.random.normal( size = (self.model[0].DoF, self.npts) )       
        self.ptp = {}             
        
        # Risk tolerance
        self.riskTol = 5e-2
        
        # Initialize changepoints
        self.changepoints = {}
    
    def apply(self):            
        for t_ in range(self.nData):
            print('Time:', t_)
            
            # Sample from posterior run length
            self.getRunLengthSamples(t_)      
            
            # Update models and samplers
            self.data2t = self.data[:t_]
            self.updateInferenceModels(t_)
            
            # Get predictive samples
            self.getPredictiveSamples(t_)
            
            # Get predictive statistics
            self.getPredictiveStats(t_)
            
            # Observe new data point
            datat = self.data[t_]
            
            # Check whether changepoint            
            self.checkIfChangepoint(t_, datat)
            
            # Get run length posterior           
            self.getRunLengthProbability(t_, datat)
        
    def getRunLengthSamples(self, t_):
        if t_ != 0:
            rld = stats.rv_discrete( values = ( np.arange( min(t_+1, self.rmax) ), self.rlp ) )
            self.rls = rld.rvs( size = self.nrls )
        else:
            self.rls = np.zeros(self.nrls)
            
    def updateInferenceModels(self, t_):
        if t_ != 0:
            self.trmax = min(t_ + 1, self.rmax)
            with multiprocessing.Pool(40) as pool:
                results = pool.map( self.updateInferenceModels2Pool, range(1, self.trmax) )
                for r_ in range(1, self.trmax): 
                    self.model[r_] = results[r_-1][0]
                    self.smc[r_]   = results[r_-1][1]
                    self.pts[r_]   = self.smc[r_].particles
                
    def updateInferenceModels2Pool(self, r_):
        if r_ < self.trmax - 1:
            model = HAWKES_SMC(self.data2t[-r_:], self.data2t[-r_-1])
        else:
            model = HAWKES_SMC(self.data2t[-r_:], 0)
        smc = SMC(model)
        smc.apply()
        return (model, smc)
    
    def getPredictiveSamples(self, t_):
        self.pps = np.array([])     
        r_vals, r_counts = np.unique(self.rls, return_counts = 1) 
        for k_ in range( len(r_vals) ): # PARALLELIZABLE!
            r_val = r_vals[k_]
            r_count = r_counts[k_]          
            thetas = self.pts[r_val][:, np.random.choice(np.arange(self.npts), r_count)]
            if t_ == 0:
                tmp = self.model[r_val].simulateNewEvent(thetas)
            else:
                tmp = self.data[t_-1] + self.model[r_val].simulateNewEvent(thetas)
            self.pps = np.hstack( ( self.pps, tmp ) )
       
    def getPredictiveStats(self, t_):   
        self.pred_mean[t_] = np.mean(self.pps)
        
        if self.flag_lci == 1:
            self.percentile_l[t_] = np.percentile(self.pps, self.risk_level_l)        
        if self.flag_rci == 1:
            self.percentile_r[t_] = np.percentile(self.pps, 100 - self.risk_level_r)
                        
    def getRunLengthProbability(self, t_, datat):   
        pp0 = np.mean( stats.expon.pdf(datat, scale = np.exp( - self.pts[0][0,:] ) ) )
        pp = np.hstack( (pp0, np.zeros( min(t_, self.rmax-1) )) )
        for r_ in range(1, min(t_, self.rmax)): # PARALLELIZABLE!
            pp[r_] = np.mean( np.exp( - self.model[r_].getPredMinusLogLikelihood( self.pts[r_], datat ) ) )
        if  t_ < self.rmax-1:
            if len(pp)==1:
                log_predictpp = np.maximum(-500, np.log(pp[-1]))
                self.predictprob.append(log_predictpp)
            else:
                log_predictpp = np.maximum(-500, np.log(pp[-2]))
                self.predictprob.append(log_predictpp)
        else:
            log_predictpp = np.maximum(-500, np.log(pp[-1]))
            self.predictprob.append(log_predictpp)


        # Calculate run length posterior
        jppp = self.jp * pp                            # Joint x predictive prob
        gp = jppp * ( 1 - self.H )                     # Growth prob
        cp = np.sum( jppp * self.H )                   # Changepoint prob
        self.jp  = np.hstack( (cp, gp) )[:self.rmax]   # Joint prob
        ep = np.sum( self.jp )                         # Evidence prob
        self.rlp = self.jp / ep                        # Run length posterior
                
    def checkIfChangepoint(self, t_, datat):
        bool_test = 0
        if self.t0[t_ - 1]>30 and t_>30:
            if self.flag_lci:           
                if datat < self.percentile_l[t_]: 
                    risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                    print("l",risk)
                    if risk > self.riskTol:
                        self.changepoints.update( {t_: risk} )   
                        print('Changepoint at time', t_, 'for drastic decrease')
                        bool_test = 1
                    
            if self.flag_rci: 
                if datat > self.percentile_r[t_]: 
                    risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
                    print("r",risk)
                    if risk > self.riskTol:
                        self.changepoints.update( {t_: risk} )   
                        print('Changepoint at time', t_, 'for drastic increase')
                        bool_test = 1
        if bool_test == 1:
                if t_ == 0:
                    self.t0 = [0]
                else:
                    self.t0.append(0)
        else:
            if t_ == 0:
                self.t0 = [0]
            else:
                self.t0.append(self.t0[-1] + 1) 