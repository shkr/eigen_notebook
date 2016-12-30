import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import scipy.stats as stats
import scipy
import seaborn as sns

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CLRM')

class Result(object):
    def __init__(self, files, model, trace):
        self.files = files
        self.model = model
        self.trace = trace
        
class CLRM(object):
    
    
    def __init__(self, file_name, x, y, model, trace, ppc):
        self.file_name = file_name
        self.x, self.y = x, y
        self.model = model
        self.trace = trace
        self.ppc = ppc

    @staticmethod    
    def plot_data(file_name, x, y):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        ax.scatter(x[:], y[:], s=10, c='g')
        ax.set_ylabel("Y")
        ax.set_xlabel('X')
        ax.set_title("training_data [{}]".format(file_name))
        ax.legend()

        
    @staticmethod
    def xy_op(file_name, scaler=None):

        logger.info("reading file {}".format(file_name))
        df = pd.read_csv(file_name)
        exog = df['x'].as_matrix()
        endo = df['y'].as_matrix()

        sample_size = exog.shape[0]

        x = exog
        logger.info("shape of [X] is {}".format(x.shape))
        logger.debug("[X] is {}".format(x[:3]))
        
        y = endo
        logger.info("shape of [Y] is {}".format(y.shape))
        logger.debug("[Y] is {}".format(y[:3]))
        
        from sklearn import preprocessing
        if scaler is 'RobustScaler':
            x_scaler = preprocessing.RobustScaler().fit(x)
            y_scaler = preprocessing.RobustScaler().fit(endo)
        
            return (x_scaler.transform(x), y_scaler.transform(y))
        else:
            return (x, y)
    
    @staticmethod
    def hierarchical_model(files=None):
        
        # prepare dataset
        if files is None:
            files = ["data_1_1.csv", "data_1_2.csv", "data_1_3.csv", "data_1_4.csv", "data_1_5.csv"]
            data = [CLRM.xy_op(file_name) for file_name in files]
            data.append((data[0][0][50:], data[0][1][50:]))
            data.append((data[1][0][50:], data[1][1][50:]))
            data[0] = (data[0][0][:50], data[0][1][:50])
            data[1] = (data[1][0][:50], data[1][1][:50])
            shape = 7
        else:
            data = [CLRM.xy_op(file_name) for file_name in files]
            shape = len(files)
          
        x = np.vstack([e[0] for e in data]).T
        y = np.vstack([e[1] for e in data]).T
        
        
        with pm.Model() as model:
            
            # Hyperpriors for group nodes
            mu_a = pm.Normal('mu_alpha', mu=0., sd=100**2)
            sigma_a = pm.Uniform('sigma_alpha', lower=0, upper=100)
            mu_b = pm.Normal('mu_beta', mu=0., sd=100**2)
            sigma_b = pm.Uniform('sigma_beta', lower=0, upper=100)

            # Intercept for each county, distributed around group mean mu_a
            # Above we just set mu and sd to a fixed value while here we
            # plug in a common group distribution for all a and b (which are
            # vectors of length n_counties).
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=shape)
            # Intercept for each county, distributed around group mean mu_a
            b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=shape)
            
            # Hyperpriors for the error node
            w_min, w_max = 0.0, 0.5
            nu_min, nu_max = 1.0, 10.0
            
            # Model error
            w = pm.Uniform("w", lower=w_min, upper=w_max)
            nu = pm.Uniform("nu", lower=nu_min, upper=nu_max)

            # Model prediction
            mu = (a*x) + (b)
            # Observed 
            y_obs = pm.StudentT("y_obs", mu=mu, lam=w/np.abs(x), nu=nu, observed=y)
            
        return model
    
    @staticmethod
    def model(x, y):
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
        with pm.Model() as model:

            # Define priors
            b = pm.Uniform("b", lower=-50, upper=50)
            a = pm.Uniform("a", lower=-50, upper=50)
            w = pm.Uniform("w", lower=0.0, upper=0.5)
            nu = pm.Uniform("nu", lower=1.0, upper=10.0)

            # Identity Link Function 
            mu = b + a*x
            y_obs = pm.StudentT("y_obs", mu=mu, lam=w/np.abs(x), nu=nu, observed=y)
            
        return model
    
    @staticmethod
    def mcmc(model, number_of_samples=5000):
        with model:
            # Use ADVI for initialization
            mu, sds, elbo = pm.variational.advi(n=100000)
            step = pm.NUTS(scaling=model.dict_to_array(sds)**2, is_cov=True)
            trace = pm.sample(number_of_samples, step, start=mu)       
        return trace
    
    @staticmethod
    def result_and_diagnostics(results):
        
        for result in results:
            
            print("Results and Diagnostics for {}".format(result.file_name))

            # plot the trace
            pm.summary(result.trace)
            pm.traceplot(result.trace)

            # plot the ppc
            """
            Posterior Predictive Distribution
            contains 10000 generated data sets (containing 100 samples each), 
            each using a different parameter setting from the posterior
            """
            ppc = pm.sampling.sample_ppc(result.trace, model=result.model)

        
            # posterior predictive mean
            def plot_posterior_predictive_mean(ax):
                # Posterior Predictive of the mean
                sns.distplot([np.mean(n) for n in ppc['y_obs']], kde=False, ax=ax)
                ax.axvline(y.mean())
                ax.set(title='Posterior predictive of the mean', xlabel='mean(y_obs)', ylabel='Frequency');
                return ax

            # posterior cdf
            def plot_posterior_cdf(ax):
                sns.distplot(y,
                     hist_kws=dict(cumulative=True),
                     kde_kws=dict(cumulative=True),
                     ax=ax)
                sns.distplot(y,
                     hist_kws=dict(cumulative=True),
                     kde_kws=dict(cumulative=True),
                     ax=ax)

                # Posterior Predictive of the mean
                sns.distplot([n.mean() for n in ppc['y_obs']], kde=False, ax=ax)
                ax.axvline(y.mean())
                ax.set(title='Posterior predictive of the mean', xlabel='mean(y_obs)', ylabel='Frequency');


            # Linear Fit
            def plot_linear_fit(ax):
                # Select Best Estimate
                b_mean = np.mean([item['b'] for item in trace])
                a_mean = np.mean([item['a'] for item in trace])
                ax.scatter(x[:], y[:], s=10, c='g')
                ax.plot(x[:], (b_mean + a_mean*x)[:])
                ax.set_ylabel("Y")
                ax.set_xlabel('X')
                ax.set_title("training_data [{}] with linear fit".format(file_name))
                ax.legend() 


            fig = plt.figure(figsize=(10, 10))
            plot_posterior_predictive_mean(fig.add_subplot(211))
            plot_linear_fit(fig.add_subplot(212))
            fig.show() 