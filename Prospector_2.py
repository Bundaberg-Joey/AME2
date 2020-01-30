#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:39:05 2020

@author: Hook
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import GPy
from scipy.stats import norm

class prospector2(object):

    def __init__(self, X,costs,acquisition_function='Thompson'):
        """ Initializes by storing all feature values """
        self.X = X
        self.n, self.d = X.shape
        self.update_counter = 10
        self.updates_per_big_fit = 10
        self.estimate_tau_counter = 10
        self.tau_update = 10
        self.acquisition_function=acquisition_function
        self.y_max = None    
        self.Thompson_p_expensive=0.1 # probability of choosing expensive test when using Thompson sampling 
        self.costs=costs


    """ incorporates new data """
    """ every 10 turns also fits hyperparameters and inducing points """
    def fit(self,Y,STATUS,ntop=100,nrecent=100,nmax=400,ntopmu=100,ntopvar=100,nkmeans=300,nkeamnsdata=5000,lam=1e-4,mz=10,mk=1000,max_attempts=5):
        """ put everything in the format we need for rest of this function """
        uu=[i for i in range(self.n) if STATUS[i]==0]
        tu=[i for i in range(self.n) if STATUS[i] in [2,3]]
        tt=[i for i in range(self.n) if STATUS[i]==4]
        z=Y[:,0]
        y=Y[:,1]
        X=self.X
        self.zdata=[tu+tt,z[tu+tt]]
        self.ydata=[tt,y[tt]]
        # each time we spend 10 we update the hyperparameters, otherwise we just update the data which is a lot faster
        if self.update_counter>=self.updates_per_big_fit:
            """ upto max_attempts retries if fitting fails """
            done=False
            attempts=0
            while done==False and attempts<max_attempts:
                try:
                    """ fit a GPy dense model to get hyperparameters """
                    """ take subsample for tested data for fitting """
                    """ ntop top samples """
                    """ nrecent most recent samples """
                    """ random samples up to total of nmax """
                    self.n,self.d=X.shape
                    """ do z scores first """
                    print('fitting z hyperparameters')
                    ntt=len(tt)
                    ntu=len(tu)
                    if ntt+ntu>nmax:
                        toptt=[tt[i] for i in np.argsort(y[tt])[-ntop:]]
                        recenttt=tt[-int(nrecent/2):]
                        recenttu=tu[-int(nrecent/2):]
                        nrand=nmax-len(toptt)-len(recenttt)-len(recenttu)
                        randtt=list(np.random.choice(tt,min(int(nrand/2),ntt),False))
                        randtu=list(np.random.choice(tu,min(int(nrand/2),ntu),False))
                        samplett=list(set(toptt+recenttt+randtt))
                        sampletu=list(set(recenttu+randtu))
                        ztrainy=z[samplett+sampletu]
                        ztraini=samplett+sampletu
                    else:
                        ztraini=tu+tt
                        ztrainy=z[tu+tt]
                    """ fit dense GPy model to this data """
                    mfz=GPy.mappings.Constant(input_dim=self.d,output_dim=1)
                    kz = GPy.kern.RBF(self.d,ARD=True,lengthscale=np.ones(self.d))
                    self.GPz = GPy.models.GPRegression(X[ztraini],ztrainy.reshape(-1,1),kernel=kz,mean_function=mfz)
                    try:
                        self.GPz.optimize('bfgs')
                    except:
                        try:
                            print('failed fitting z hyperparameters - trying again in safe mode')
                            mfz=GPy.mappings.Constant(input_dim=self.d,output_dim=1)
                            kz = GPy.kern.RBF(self.d,ARD=True,lengthscale=np.ones(self.d))
                            self.GPz = GPy.models.GPRegression(X[ztraini],ztrainy.reshape(-1,1),kernel=kz,mean_function=mfz)
                            self.GPz.Gaussian_noise.variance=np.cov(ztrainy)
                            self.GPz.optimize('bfgs')
                        except:
                            print('failed fitting z hyperparameters in safe mode - aborting')
                    self.muz=self.GPz.flattened_parameters[0]
                    self.az=self.GPz.flattened_parameters[1]
                    self.lz=self.GPz.flattened_parameters[2]
                    self.bz=self.GPz.flattened_parameters[3]
                    """ now do y scores """
                    print('fitting y hyperparameters')
                    if ntt>nmax:
                        ytrainy=y[samplett]
                        ytraini=samplett
                        ytrainz=z[samplett]
                    else:
                        ytraini=tt
                        ytrainy=y[tt]
                        ytrainz=z[tt]
                    """ fit dense GPy model to this data """
                    mfy=GPy.mappings.Constant(input_dim=self.d+1,output_dim=1)
                    ky = GPy.kern.RBF(self.d+1,ARD=True,lengthscale=np.ones(self.d+1))
                    self.GPy = GPy.models.GPRegression(np.hstack((X[ytraini],ytrainz.reshape(-1,1))),ytrainy.reshape(-1,1),kernel=ky,mean_function=mfy)
                    try:
                        self.GPy.optimize('bfgs')
                    except:
                        try:
                            print('failed fitting y hyperparameters - trying again in safe mode')
                            mfy=GPy.mappings.Constant(input_dim=self.d+1,output_dim=1)
                            ky = GPy.kern.RBF(self.d+1,ARD=True,lengthscale=np.ones(self.d+1))
                            self.GPy = GPy.models.GPRegression(np.hstack((X[ytraini],ytrainz.reshape(-1,1))),ytrainy.reshape(-1,1),kernel=ky,mean_function=mfy)
                            self.GPy.Gaussian_noise.variance=np.cov(ytrainy)
                            self.GPy.optimize('bfgs')
                        except:
                            print('failed fitting y hyperparameters in safe mode - aborting')
                    self.muy=self.GPy.flattened_parameters[0]
                    self.ay=self.GPy.flattened_parameters[1]
                    self.ly=self.GPy.flattened_parameters[2]
                    self.by=self.GPy.flattened_parameters[3]
                    """ now pick inducing points for sparse model """
                    """ use all the train points as above """
                    """ ntopmu most promising untested points """
                    """ ntopvar most uncertain untested points """
                    """ nkmeans cluster centers from untested data """
                    print('selecting z inducing points')
                    self.pz=self.GPz.predict(X)
                    self.pynaive=self.GPy.predict(np.hstack((X,self.pz[0])))
                    topmuuu=[uu[i] for i in np.argsort(self.pynaive[0][uu].reshape(-1))[-int(ntopmu/2):]]
                    topmutu=[tu[i] for i in np.argsort(self.pynaive[0][tu].reshape(-1))[-int(ntopmu/2):]]
                    topvarzuu=[uu[i] for i in np.argsort(self.pz[1][uu].reshape(-1))[-ntopvar:]]
                    znystrom=list(set(topmuuu+topmutu+topvarzuu+ztraini))
                    kms=KMeans(n_clusters=nkmeans,max_iter=5).fit(np.divide(X[list(np.random.choice(uu,nkeamnsdata))],self.lz)) 
                    self.Mz=np.vstack((X[znystrom],np.multiply(kms.cluster_centers_,self.lz)))
                    """ fit sparse model """
                    """ lam controls jitter in g samlpes """
                    print('fitting sparse z model')
                    DXMz=euclidean_distances(np.divide(X,self.lz),np.divide(self.Mz,self.lz),squared=True)
                    self.SIG_XMz=self.az*np.exp(-DXMz/2)
                    DMMz=euclidean_distances(np.divide(self.Mz,self.lz),np.divide(self.Mz,self.lz),squared=True)
                    self.SIG_MMz=self.az*np.exp(-DMMz/2)+np.identity(self.Mz.shape[0])*lam*self.az
                    self.Bz=self.az+self.bz-np.sum(np.multiply(np.linalg.solve(self.SIG_MMz,self.SIG_XMz.T),self.SIG_XMz.T),0)
                    K=np.matmul(self.SIG_XMz[tu+tt].T,np.divide(self.SIG_XMz[tu+tt],self.Bz[tu+tt].reshape(-1,1)))
                    self.SIG_MM_posz=self.SIG_MMz-K+np.matmul(K,np.linalg.solve(K+self.SIG_MMz,K))+lam*self.az     
                    self.SIG_MM_posz=0.5*(self.SIG_MM_posz+self.SIG_MM_posz.T)
                    J=np.matmul(self.SIG_XMz[tu+tt].T,np.divide(z[tu+tt]-self.muz,self.Bz[tu+tt]))
                    self.mu_M_posz=self.muz+J-np.matmul(K,np.linalg.solve(K+self.SIG_MMz,J))
                    self.vz=np.linalg.solve(self.SIG_MMz,self.mu_M_posz-self.muz)
                    self.Vz=np.linalg.solve(self.SIG_MMz,np.linalg.solve(self.SIG_MMz,self.SIG_MM_posz).T)
                    print('selecting y inducing points')
                    topmutu=[tu[i] for i in np.argsort(self.pynaive[0][tu].reshape(-1))[-ntopmu:]]
                    topvarztu=[tu[i] for i in np.argsort(self.pynaive[1][tu].reshape(-1))[-ntopvar:]]
                    ynystrom=topmutu+topvarztu+ytraini
                    zsamples=self.samplesz(nsamples=mz)
                    p=np.random.permutation(self.n)
                    kms=KMeans(n_clusters=nkmeans,max_iter=5).fit(np.divide(np.vstack(([np.hstack((X[p[mk*i:mk*(i+1)]],zsamples[p[mk*i:mk*(i+1)],i].reshape(-1,1))) for i in range(mz)])),self.ly))
                    self.My=np.vstack((np.hstack((X[ynystrom],z[ynystrom].reshape(-1,1))),np.multiply(kms.cluster_centers_,self.ly)))
                    """ fit sparse model """
                    """ lam controls jitter in g samlpes """
                    print('fitting sparse y model')
                    ADXMy=euclidean_distances(np.divide(X,self.ly[:-1]),np.divide(self.My[:,:-1],self.ly[:-1]),squared=True)
                    self.ASIG_XMy=self.ay*np.exp(-ADXMy/2)
                    DMMy=euclidean_distances(np.divide(self.My,self.ly),np.divide(self.My,self.ly),squared=True)
                    self.SIG_MMy=self.ay*np.exp(-DMMy/2)+np.identity(self.My.shape[0])*lam*self.ay
                    self.SIG_XMytt=np.multiply(np.exp(-euclidean_distances(np.divide(z[tt].reshape(-1,1),self.ly[-1]),np.divide(self.My[:,-1].reshape(-1,1),self.ly[-1]),squared=True)/2),self.ASIG_XMy[tt])
                    self.Bytt=self.ay+self.by-np.sum(np.multiply(np.linalg.solve(self.SIG_MMy,self.SIG_XMytt.T),self.SIG_XMytt.T),0) 
                    K=np.matmul(self.SIG_XMytt.T,np.divide(self.SIG_XMytt,self.Bytt.reshape(-1,1)))
                    self.SIG_MM_posy=self.SIG_MMy-K+np.matmul(K,np.linalg.solve(K+self.SIG_MMy,K))+lam*self.ay       
                    self.SIG_MM_posy=0.5*(self.SIG_MM_posy+self.SIG_MM_posy.T)
                    J=np.matmul(self.SIG_XMytt.T,np.divide(y[tt]-self.muy,self.Bytt))
                    self.mu_M_posy=self.muy+J-np.matmul(K,np.linalg.solve(K+self.SIG_MMy,J))
                    self.vy=np.linalg.solve(self.SIG_MMy,self.mu_M_posy-self.muy)
                    self.Vy=np.linalg.solve(self.SIG_MMy,np.linalg.solve(self.SIG_MMy,self.SIG_MM_posy).T)
                    """ test model by sampling from y """
                    """ if there is a poblem with the model this should cause a crash """
                    print('test sample')
                    self.samplesy(nsamplesz=1,nsamplesy=1)
                    done=True
                    self.update_counter=0
                except:
                    attempts=attempts+1
        else:
            self.zdata=[tu+tt,z[tu+tt]]
            self.ydata=[tt,y[tt]]
            print('updating sparse z model')
            K=np.matmul(self.SIG_XMz[tu+tt].T,np.divide(self.SIG_XMz[tu+tt],self.Bz[tu+tt].reshape(-1,1)))
            self.SIG_MM_posz=self.SIG_MMz-K+np.matmul(K,np.linalg.solve(K+self.SIG_MMz,K))+lam*self.az    
            self.SIG_MM_posz=0.5*(self.SIG_MM_posz+self.SIG_MM_posz.T)
            J=np.matmul(self.SIG_XMz[tu+tt].T,np.divide(z[tu+tt]-self.muz,self.Bz[tu+tt]))
            self.mu_M_posz=self.muz+J-np.matmul(K,np.linalg.solve(K+self.SIG_MMz,J))
            self.vz=np.linalg.solve(self.SIG_MMz,self.mu_M_posz-self.muz)
            self.Vz=np.linalg.solve(self.SIG_MMz,np.linalg.solve(self.SIG_MMz,self.SIG_MM_posz).T)
            print('updating sparse y model')
            self.SIG_XMytt=np.multiply(np.exp(-euclidean_distances(np.divide(z[tt].reshape(-1,1),self.ly[-1]),np.divide(self.My[:,-1].reshape(-1,1),self.ly[-1]),squared=True)/2),self.ASIG_XMy[tt])
            self.Bytt=self.ay+self.by-np.sum(np.multiply(np.linalg.solve(self.SIG_MMy,self.SIG_XMytt.T),self.SIG_XMytt.T),0)
            K=np.matmul(self.SIG_XMytt.T,np.divide(self.SIG_XMytt,self.Bytt.reshape(-1,1)))
            self.SIG_MM_posy=self.SIG_MMy-K+np.matmul(K,np.linalg.solve(K+self.SIG_MMy,K))+lam*self.ay              
            self.SIG_MM_posy=0.5*(self.SIG_MM_posy+self.SIG_MM_posy.T)
            J=np.matmul(self.SIG_XMytt.T,np.divide(y[tt]-self.muy,self.Bytt))
            self.mu_M_posy=self.muy+J-np.matmul(K,np.linalg.solve(K+self.SIG_MMy,J))
            self.vy=np.linalg.solve(self.SIG_MMy,self.mu_M_posy-self.muy)
            self.Vy=np.linalg.solve(self.SIG_MMy,np.linalg.solve(self.SIG_MMy,self.SIG_MM_posy).T)
        
        
        """ 
        key attributes updated by fit 
        
        stuff to do with cheap scores z 
        
        self.SIG_XMz : prior covarience matrix between data and inducing points
        
        self.SIG_MMz : prior covarience matrix at inducing points
        
        self.SIG_MM_posz : posterior covarience matrix at inducing points
        self.mu_M_posz : posterior mean at inducing points
        
        self.vz : combination of the above that get used a lot
        self.Vz : combination of the above that get used a lot
        
        stuff to do with expensive scores y
        
        self.ASIG_XMy : contribution to prior covarience matrix between data and inducing points from features

        Everything else is the same as for z but with a y at the end
      
        """
        
    """ get z prediction on all points given to fit """        
    def predictz_all(self):
        mu_X_posz=self.muz+np.matmul(self.SIG_XMz,self.vz)
        var_X_posz=np.sum(np.multiply(np.matmul(self.Vz,self.SIG_XMz.T),self.SIG_XMz.T),0)
        return mu_X_posz,var_X_posz
    
    """ get y prediction on all points given to fit subject to conditional z """        
    def predicty_condz_all(self,zc,lam=1e-6):
        SIG_XMy_queary=np.multiply(np.exp(-euclidean_distances(np.divide(zc.reshape(-1,1),self.ly[-1]),np.divide(self.My[:,-1].reshape(-1,1),self.ly[-1]),squared=True)/2),self.ASIG_XMy)
        mu_X_posy_queary=self.muy+np.matmul(SIG_XMy_queary,self.vy)
        var_X_posy_queary=np.sum(np.multiply(np.matmul(self.Vy,SIG_XMy_queary.T),SIG_XMy_queary.T),0)
        return mu_X_posy_queary,var_X_posy_queary
    
    """ samples posterior on full dataset """
    def samplesz(self,nsamples=10):
        print('sampling from sparse z model')
        samples_M_posz=np.random.multivariate_normal(self.mu_M_posz,self.SIG_MM_posz,nsamples).T
        samples_X_posz=self.muz+np.matmul(self.SIG_XMz,np.linalg.solve(self.SIG_MMz,samples_M_posz-self.muz))+self.bz**0.5*np.random.randn(self.n,nsamples)
        samples_X_posz[self.zdata[0],:]=np.repeat(self.zdata[1].reshape(-1,1),nsamples,1)
        return samples_X_posz

    """ samples posterior on full dataset """
    def samplesy(self,nsamplesz=10,nsamplesy=10,withnoise=True):
        samples_X_posz=self.samplesz(nsamples=nsamplesz)
        print('sampling from sparse y model')
        samples_M_posy=np.random.multivariate_normal(self.mu_M_posy,self.SIG_MM_posy,nsamplesy).T
        C=np.linalg.solve(self.SIG_MMy,samples_M_posy-self.muy)
        samples_X_posy=np.zeros((self.n,nsamplesz*nsamplesy))
        for j in range(nsamplesz):
            samples_X_posy[:,j*nsamplesy:(j+1)*nsamplesy]=self.muy+np.matmul(np.multiply(np.exp(-euclidean_distances(np.divide(samples_X_posz[:,j].reshape(-1,1),self.ly[-1]),np.divide(self.My[:,-1].reshape(-1,1),self.ly[-1]),squared=True)/2),self.ASIG_XMy),C)
        if withnoise:
            samples_X_posy=samples_X_posy+self.by**0.5*np.random.randn(self.n,nsamplesz*nsamplesy)
        samples_X_posy[self.ydata[0],:]=np.repeat(self.ydata[1].reshape(-1,1),nsamplesz*nsamplesy,1)
        return samples_X_posz,samples_X_posy
    
    def Thompson_Sample(self,STATUS):
        alpha=self.samplesy(nsamplesz=1,nsamplesy=1,withnoise=False)[1]
        uu=[i for i in range(self.n) if STATUS[i]==0]
        iuu=uu[np.argmax(alpha[uu])]
        tu=[i for i in range(self.n) if STATUS[i]==2]
        if len(tu)>0:
            itu=tu[np.argmax(alpha[tu])]
            if np.random.rand()<self.Thompson_p_expensive:
                return itu,1
        return iuu,0

    def pick_next(self, STATUS, N=100, nysamples=100):
        """
        Picks next material to sample

        :param STATUS: np.array(), keeps track of which materials have been assessed / what experiments conducted
        :param acquisition_function: The sampling method to be used by the AMI to select new materials
        :param N: The number of materials which the `Greedy N` algorithm is attempting to optimise for
        :param nysamples: Number of samples to draw from posterior for Greedy N optimisation

        :return: ipick: int, the index value in the feature matrix `X` for non-tested materials
        """
        if self.acquisition_function == 'Thompson':
            ipick,exppick=self.Thompson_Sample(STATUS)
        else:
            # if no valid acquisition_function entered then pick at random 
            ipick = np.random.choice([i for i in range(self.n) if STATUS[i] in [0,2]])
            exppick=0
            print('enter a valid acquisition function - picking randomly')
        
        self.update_counter+=self.costs[0]
            
        return ipick,exppick
        
    
    
    
    