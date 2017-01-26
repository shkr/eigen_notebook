import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import scipy.stats as stats
import scipy
import seaborn as sns
import time
import logging
import random
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CLRM')

class Result(object):
    
    def __init__(self, 
                 file_name, 
                 model, 
                 trace):
        self.file_name = file_name
        self.model = model
        self.trace = trace
        self.mean_estimates = {}
        self.stddev_estimates = {}

        keys = trace[0].keys()
        for k in keys:
            self.mean_estimates[k] = np.mean([t[k] for t in trace])
            self.stddev_estimates[k] = np.std([t[k] for t in trace])
            
class CLRM(object):
    
    def __init__(self, file_name, variance_fn=None):
        self.file_name = file_name
        self.x, self.y = self.xy_op(file_name)
        self.model = None
        self.trace = None
        self.ppc = None
        self.result = None
        self.variance_fn = lambda w, x: w/np.abs(x) or variance_fn
    
    def generate_ppc(self, plot_ppc=False):
        if self.trace is None or self.model is None:
            raise ValueError('model must be initialized and trace must be generated')
        self.ppc = pm.sample_ppc(trace=self.trace, model=self.model)
        if plot_ppc:
            ax = plt.subplot()
            x, y = self.x, self.y
            y_ppc = np.array([random.choice(e) for e in ppc['y_obs'].T])
            ax.scatter(x[:], y_ppc, s=10, c='g')
            ax.scatter(x[:], y, s=10, c='b')
        return self.ppc
    
    def fit_linear_model(self, number_of_samples=5000):
        # load class variables
        file_name = self.file_name
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        """
        Student T distribution

        This one with uniform priors

        Parameters
            ----------
            nu : int
                Degrees of freedom (nu > 0).
            mu : float
                Location parameter.
            lam : float
                Scale parameter (lam > 0).

            The (ei) are independent noise from a distribution that depends on x 
            as well as on global parameters; however, the noise distribution has 
            conditional mean zero given x                    

            Hypothesis

            ei is drawn from a Student T distribution. the sclae parameter for ei is x.
            The global parameter for ei is nu or degrees of freedom

            How can you change the scale of a standard student-t distribution
        """
        logger.info("Fitting linear model for filename {}".format(file_name))       
        model = pm.Model()
        start_time = time.time()
        with model:
            # Define priors
            b = pm.Normal("b", mu=0., sd=100**2)
            a = pm.Normal("a", mu=0., sd=100**2)
            w = pm.Uniform("w", lower=0.0, upper=10.0)
            nu = pm.Uniform("nu", lower=1.0, upper=10.0)

            # Identity Link Function 
            mu = b + a*x
            y_obs = pm.StudentT("y_obs", mu=mu, lam=variance_fn(w, x), nu=nu, observed=y)
        
        with model:
            # Use ADVI_MAP for initialization
            trace = pm.sample(number_of_samples, init='advi_map')
            
        logger.info("Time taken = {}".format(time.time()-start_time))
        
        # Set class variables
        self.model = model
        self.trace = trace
        self.result = Result(file_name, model, trace)
        return self

    
    def plot_input_data(self):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        ax.scatter(self.x[:], self.y[:], s=10, c='g')
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}]".format(self.file_name))
        ax.legend()
    
    
    def rmsd(self):
        """
        Calculates the RMSD for the given point estimates
        and training data
        """
        if self.result is None:
            raise ValueError('Linear Model must be fit to calculate RMSD')
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        result = self.result
        ppc = self.ppc or self.generate_ppc()
        a_est  = result.mean_estimates['a']
        b_est  = result.mean_estimates['b']
        w_est  = result.mean_estimates['w']
        nu_est = result.mean_estimates['nu']
        
        rmsd_ = 0
        y_ppc = ppc['y_obs'].T
        print(y_ppc.shape)
        for y_obs, y_ppc_sample in zip(y, y_ppc):
            closest_y_delta = min([np.abs(e-y_obs) for e in y_ppc_sample])
            rmsd_ += np.power(closest_y_delta, 2)
        rmsd_ /= len(y_ppc)
        rmsd_ = np.power(rmsd_, 0.5)
        return rmsd_
    
    def result_and_diagnostics(self):
        
        logger.info("Results for {}".format(self.file_name))
        
        if self.result is None:
            raise ValueError('Linear Model must be fit for printing result')
        
        # load class objects
        file_name = self.file_name
        x, y = self.x, self.y
        result = self.result
        variance_fn = self.variance_fn
        a_est  = result.mean_estimates['a']
        b_est  = result.mean_estimates['b']
        w_est  = result.mean_estimates['w']
        nu_est = result.mean_estimates['nu']

        print('MAD = {}'.format(self.mad()))
        print('RMSD = {}'.format(self.rmsd()))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for item in result.trace[:50:]:
            a_p = item['a']
            b_p = item['b']
            ax.plot(x[:], (b_p + a_p*x)[:], 'g', alpha=1, lw=.01)
        ax.scatter(x[:], y[:], alpha=1, color='k', marker='.', s=50, label='original data')    
        ax.set_ylim(-20, +20)
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}] with linear fit".format(file_name))        
        
    @staticmethod
    def xy_op(file_name):

        logger.info("reading file {}".format(file_name))
        df = pd.read_csv(file_name)
        exog = df['x'].as_matrix()
        endo = df['y'].as_matrix()

        sample_size = exog.shape[0]

        x = exog
        logger.info("shape of X is {}".format(x.shape))
        logger.debug("X is {}".format(x[:3]))
        
        y = endo
        logger.info("shape of Y is {}".format(y.shape))
        logger.debug("Y is {}".format(y[:3]))
        
        return (x, y)            
class CLRM(object):
    
    def __init__(self, file_name, variance_fn=None):
        self.file_name = file_name
        self.x, self.y = self.xy_op(file_name)
        self.model = None
        self.trace = None
        self.ppc = None
        self.result = None
        self.variance_fn = lambda w, x: w/np.abs(x) or variance_fn
    
    def generate_ppc(self, plot_ppc=False):
        if self.trace is None or self.model is None:
            raise ValueError('model must be initialized and trace must be generated')
        self.ppc = pm.sample_ppc(trace=self.trace, model=self.model)
        if plot_ppc:
            ax = plt.subplot()
            x, y = self.x, self.y
            y_ppc = np.array([random.choice(e) for e in ppc['y_obs'].T])
            ax.scatter(x[:], y_ppc, s=10, c='g')
            ax.scatter(x[:], y, s=10, c='b')
        return self.ppc
    
    def fit_linear_model(self, number_of_samples=5000):
        # load class variables
        file_name = self.file_name
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        """
        Student T distribution

        This one with uniform priors

        Parameters
            ----------
            nu : int
                Degrees of freedom (nu > 0).
            mu : float
                Location parameter.
            lam : float
                Scale parameter (lam > 0).

            The (ei) are independent noise from a distribution that depends on x 
            as well as on global parameters; however, the noise distribution has 
            conditional mean zero given x                    

            Hypothesis

            ei is drawn from a Student T distribution. the sclae parameter for ei is x.
            The global parameter for ei is nu or degrees of freedom

            How can you change the scale of a standard student-t distribution
        """
        logger.info("Fitting linear model for filename {}".format(file_name))       
        model = pm.Model()
        start_time = time.time()
        with model:
            # Define priors
            b = pm.Normal("b", mu=0., sd=100**2)
            a = pm.Normal("a", mu=0., sd=100**2)
            w = pm.Uniform("w", lower=0.0, upper=10.0)
            nu = pm.Uniform("nu", lower=1.0, upper=10.0)

            # Identity Link Function 
            mu = b + a*x
            y_obs = pm.StudentT("y_obs", mu=mu, lam=variance_fn(w, x), nu=nu, observed=y)
        
        with model:
            # Use ADVI_MAP for initialization
            trace = pm.sample(number_of_samples, init='advi_map')
            
        logger.info("Time taken = {}".format(time.time()-start_time))
        
        # Set class variables
        self.model = model
        self.trace = trace
        self.result = Result(file_name, model, trace)
        return self

    
    def plot_input_data(self):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        ax.scatter(self.x[:], self.y[:], s=10, c='g')
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}]".format(self.file_name))
        ax.legend()
    
    
    def mad(self):
        """
        Calculates the Median Absolute Deviation for the given point estimates
        and training data
        """
        if self.result is None:
            raise ValueError('Linear Model must be fit to calculate RMSD')
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        result = self.result
        ppc = self.ppc or self.generate_ppc()
        
        a_est  = result.mean_estimates['a']
        b_est  = result.mean_estimates['b']
        w_est  = result.mean_estimates['w']
        nu_est = result.mean_estimates['nu']
        
        residual = []
        # y_ppc = np.array([random.choice(e) for e in ppc['y_obs'].T])
        y_ppc = ppc['y_obs'].T
        for y_obs, y_ppc_sample in zip(y, y_ppc):
            closest_y_delta = min([np.abs(e-y_obs) for e in y_ppc_sample])
            residual.append(closest_y_delta)        
        return np.median(residual)
    
    def rmsd(self):
        """
        Calculates the RMSD for the given point estimates
        and training data
        """
        if self.result is None:
            raise ValueError('Linear Model must be fit to calculate RMSD')
        x, y = self.x, self.y
        variance_fn = self.variance_fn
        result = self.result
        ppc = self.ppc or self.generate_ppc()
        a_est  = result.mean_estimates['a']
        b_est  = result.mean_estimates['b']
        w_est  = result.mean_estimates['w']
        nu_est = result.mean_estimates['nu']
        
        rmsd_ = 0
        y_ppc = ppc['y_obs'].T
        for y_obs, y_ppc_sample in zip(y, y_ppc):
            closest_y_delta = min([np.abs(e-y_obs) for e in y_ppc_sample])
            rmsd_ += np.power(closest_y_delta, 2)
        rmsd_ /= len(y_ppc)
        rmsd_ = np.power(rmsd_, 0.5)
        return rmsd_
    
    def result_and_diagnostics(self):
        
        logger.info("Results for {}".format(self.file_name))
        
        if self.result is None:
            raise ValueError('Linear Model must be fit for printing result')
        
        # load class objects
        file_name = self.file_name
        x, y = self.x, self.y
        result = self.result
        variance_fn = self.variance_fn
        a_est  = result.mean_estimates['a']
        b_est  = result.mean_estimates['b']
        w_est  = result.mean_estimates['w']
        nu_est = result.mean_estimates['nu']

        print('MAD = {}'.format(self.mad()))
        print('RMSD = {}'.format(self.rmsd()))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for item in result.trace[:50:]:
            a_p = item['a']
            b_p = item['b']
            ax.plot(x[:], (b_p + a_p*x)[:], 'g', alpha=1, lw=.01)
        ax.scatter(x[:], y[:], alpha=1, color='k', marker='.', s=50, label='original data')    
        ax.set_ylim(-20, +20)
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}] with linear fit".format(file_name))        
        
    @staticmethod
    def xy_op(file_name):

        logger.info("reading file {}".format(file_name))
        df = pd.read_csv(file_name)
        exog = df['x'].as_matrix()
        endo = df['y'].as_matrix()

        sample_size = exog.shape[0]

        x = exog
        logger.info("shape of X is {}".format(x.shape))
        logger.debug("X is {}".format(x[:3]))
        
        y = endo
        logger.info("shape of Y is {}".format(y.shape))
        logger.debug("Y is {}".format(y[:3]))
        
        return (x, y)