
import numpy as np 


class Estimators: 

    '''
    ================================================================================================================================
    ######################################################## DOCUMENTATION #########################################################
    ================================================================================================================================

    --------------------------------------------------------- INTRODUCTION ---------------------------------------------------------
    
    The purpose of this module is to compute different biased and unbiased estimators of the intracatable log-likelihood of a given 
    parametric distribution. The estimators computed are IAWE, SUMO, ML-SS and ML-RR (see the associated notebook for a detailed
    definition of each of these estimators).

    ---------------------------------------------------------- ATTRIBUTES ----------------------------------------------------------
    
    - z_sample: list/np.array of shape (1, n_samples); corresponding to i.i.d samples of the encoder distribution

    - sample_size: integer; size of the sample z_sample

    - theta: float; generative parameter (determining the generative model - see notebook for a clear definition of the setting)

    - phi: float; recognition parameter (determining the encoder distribution - see notebook for a clear definition of the setting)

    =================================================================================================================================
    #################################################################################################################################
    =================================================================================================================================
    '''

    def __init__(self, z_sample, theta, phi, x):
        self.z_sample = z_sample
        self.sample_size = len(z_sample)
        self.theta = theta
        self.phi = phi
        self.x = x

    def weight(x, z):
        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute importance weights according to the formula w_i = w(z_i) = p_theta(x, z_i)/q_phi(z_i|x)
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 

        - x: float ; the value of x that we use later to compute the ground truth value of the log-likelihood l_theta(x)

        - z: float ; value of the latent variable that we later replace by each value z_i in  our sample to estimate the 
             true log-likelihood value 
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------

        '''
        return joint_probability(x, z, theta)/encoder(phi, z, x) ## changer les noms des routines si c'est pas leur bon nom
    

    def l_hat(self, k, z): 
        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the biaised estimate of the log-likelihood l_theta(x) that we will later use to build our estimators. 
              Theoretically, l_hat is defined as : 1/len(z_sample) * ( w_1 + ... + w_len(z_sample) )
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 

        - k: integer; corresponding to the number of terms in the sum (or, equivalently, to the size of the samples)

        - z: np.array of shape (1, k); corresponding to an i.i.d k-sample according to the encoder's distribution
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        REMARK: 

        j'ai volontairement mis en argument de cette routine k (taille de l'échantillon) et z (échantillon) bien qu'on ait 
        en argument self.sample_size et self.z_sample car pour définir les delta_k plus tard (par exemple dans SUMO), il faut
        jouer sur la taille de l'échantillon (par exemple en faisant l_hat(k+2, ...) - l_hat(k+1, ..)
        ---------------------------------------------------------------------------------------------------------------------
        
        '''
        return np.log((1/k * sum(weight(self.x, z[i]) for i in range(1, k+ 1))))

    
    def IAWE(self): 

        ## ??

        return 
    
    
    def roulette_russe(self, I_0, Delta):

        ## prendre I_0 = 0 ??

        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the rr estimator under the conditions of theorem 1 and a geometric distribution of parameter r for the 
              number of parameters in the sum 
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 
        - I_0: float ; see the theoretical framework in the article for more details
        
        - Delta: lambda function to compute the ∆_k such as in the theoretical framework of the paper

        - r: float in (1/2, 1 - 1/(2^(1+a))) where a is such as the conditions in theorem 1 are verified (see paper)
        ---------------------------------------------------------------------------------------------------------------------
        '''
        ## !! k ranges between 0 and K included 

        ## !! we divide by P(K≥k) = (1 - r)**k when K~Geom(r)

        K = np.random.geometric(p=self.r, size=1)[0] # integer ; drawn according to a Geom(r) distribution 
                                                # corresponds to the number of terms in the sum

        return I_0 + sum(Delta(k)/((1-self.r)**k) for k in range(K + 1)) 
    

    def single_sample(self, I_0, Delta):

        ## prendre I_0 = 0 ?? 

        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the single_sample estimator (ss) under the conditions of theorem 1 and a geometric distribution of 
              parameter r for the number of parameters in the sum 
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 
        - I_0: float ; see the theoretical framework in the article for more details
        
        - Delta: lambda function to compute the ∆_k such as in the theoretical framework of the paper

        - r: float in (1/2, 1 - 1/(2^(1+a))) where a is such as the conditions in theorem 1 are verified (see paper)
        ---------------------------------------------------------------------------------------------------------------------
        '''

        K = np.random.geometric(p=self.r, size=1)[0]

        return I_0 + Delta/((1-self.r)**(K-1))*self.r


    def SUMO(self, I_0):

        ## que faire de I_0 ?

        ## QUESTION : j'ai pas compris un truc dans IAWE, on prend que les k+1 premieres valeurs dans l'échantillon ?
        ## un truc bizarre avec mon code (ci-dessous) au niveau du choix du z --> on prend pour chaque delta_k que les k+2 
        ## et k+1 premier éléments de notre échantillon ?

        Delta = lambda k: self.l_hat(k+2, self.z_sample[:k+2]) - self.l_hat(k+1, self.z_sample[:k+1])  

        return self.roulette_russe(I_0, Delta=Delta)
    
    