import numpy as np 
from scipy.stats import multivariate_normal



def joint_probability(theta, dim=20):
    """
    ---------------------------------------------------------------------------------------------------------------------
    IDÉE : génère un échantillon i.i.d de taille 'size' selon la densité jointe du vecteur aléatoire (x, z).
    ---------------------------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------------------------
    ARGUMENTS :
    - size: Taille de l'échantillon
    - dim: Dimension des vecteurs x et z ; dim = 20 car on prend x, z dans R^20
    - theta: Moyenne de la distribution N(z | theta, Id)
    ---------------------------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------------------------
    OUTPUTS :
    - Un échantillon i.i.d de taille 'size' selon la densité jointe
    ---------------------------------------------------------------------------------------------------------------------
    """
    # Génère un échantillon Z = (Z_1,...,Z_size) suivant la loi marginale précisée dans l'article
    z = np.random.multivariate_normal(np.zeros(dim) + theta, np.identity(dim), size=1)

    # Génère l'échantillon X = (X_1,...,X_20) suivant la loi conditionnelle à Z :  N(x | z, Id)
    x = np.random.multivariate_normal(z, np.eye(dim), size=1)

    return x, z

def noised_params(A, b, std_dev=0.01, dim=20):
    """
    ---------------------------------------------------------------------------------------------------------------------
    IDÉÉ : Perturber chaque composante de A et de b par une loi normale centrée et d'écart type std_dev.
    ---------------------------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------------------------
    ARGUMENTS :

    - A: Matrice dans R^{20x20}
    - b: Vecteur dans R^20
    - std_dev: Écart type de la loi normale à utiliser pour la perturbation (par défaut 0.01)
    ---------------------------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------------------------------------
    OUTPUTS :
    - noised_A: np.ndarray de shape (20,20) ; Matrice perturbée dans R^{20x20}
    - noised_b: np.ndarray de shape (1, 20) ; Vecteur perturbé dans R^20
    ---------------------------------------------------------------------------------------------------------------------
    """
    
    # Perturber chaque composante de A
    noised_A = A + np.random.normal(loc=0, scale=std_dev, size=(dim, dim))
    
    # Perturber chaque composante de b
    noised_b = b + np.random.normal(loc=0, scale=std_dev, size=dim)
    
    return noised_A, noised_b

def generate_encoder(x, k, noised_A, noised_b, dim=20): ## on oublie l'idée generate_encoder(x_sample, A, b, dim=20) --> on a une expression explicite
                                        ## de A et b 
    """
    ---------------------------------------------------------------------------------------------------------------------
    IDÉE : Génère un échantillon i.i.d. z_sample de taille k selon la loi N(noised_Ax + noised_b, 2/3 * I) dans R^20, à 
           partir d'un échantillon i.i.d. x_sample.
    ---------------------------------------------------------------------------------------------------------------------
    """
    #A, b = noised_params((1/2)*np.eye(dim), (np.zeros(20) + theta_true)/2) ## on récupère les paramètres perturbés
    #Remarque : Dans l'article on ne tire pas avec theta_true mais avec theta_hat
        
    AX_b = np.dot(x, noised_A.T) + noised_b

    cov = (2/3) * np.identity(dim) ## on calcule la variance de la normale multivariée associée à l'encodeur

    z_sample = np.random.multivariate_normal(AX_b, cov, size=k)
    z_odd = z_sample[1::2]
    z_even = z_sample[::2]

    ## POUR
    return z_sample , z_odd, z_even #AX_b #On return AX_b pour pouvoir les utiliser dans la fonction de décodage


class log_likelihood_estimators: 

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

    def __init__(self, x):
        self.x = x
        self.theta_hat = self.x.mean(axis=0)
        self.A, self.b = noised_params((1/2)*np.eye(20), (np.zeros(20) + self.theta_hat)/2)
        # self.I_0 = np.mean(self.l_hat(1, z) for z in self.z_sample)


    def weight(self, z):
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
        return (multivariate_normal.pdf(z, mean=self.theta_hat, cov=np.identity(20)) * multivariate_normal.pdf(self.x, mean=z, cov=np.identity(20))) /  multivariate_normal.pdf(self.x, mean = np.dot(self.x, self.A) + self.b, cov=(2/3)*np.identity(20))
       

    def l_hat(self, z_sample): 

        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the biaised estimate of the log-likelihood l_theta(x) that we will later use to build our estimators. 
              Theoretically, l_hat is defined as : log(1/len(z_sample) * (w_1 + ... + w_len(z_sample)))
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

        return np.log((1/(len(z_sample)) * sum(self.weight(self.x, z_sample[i]) for i in range(1, len(z_sample)+ 1))))
    
    def roulette_russe(self, I_0, Delta, K):

        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the rr estimator under the conditions of theorem 1 and a geometric distribution of parameter r for the 
              number of parameters in the sum 
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 
        - I_0: float ; see the theoretical framework in the article for more details
        
        - Delta: lambda function to compute the ∆_k such as in the theoretical framework of the paper

        - K: integer, sampled according to a law on N (positive integers)
        ---------------------------------------------------------------------------------------------------------------------
        '''

        return I_0 + sum([Delta(0)] + [Delta(k)/((1-self.r)**(k-1)) for k in range(1,K + 1)])
    

    def single_sample(self, I_0, Delta):

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

        return I_0 + Delta(K)/(((1-self.r)**(K-1))*self.r)


    def log_likelihood_SUMO(self, n_simulations):

        SUMO = []

        for _ in range(n_simulations):


            K = np.random.geometric(p=self.r, size=1)[0]

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample, _, _ = generate_encoder(self.x, K+2, noised_A=self.A, noised_b=self.B)

            ## erreur 
            ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
            Delta = lambda k: self.l_hat(z_sample[:k+2]) - self.l_hat(z_sample[:k+1])  

            I_0 = np.mean(self.l_hat(1, z) for z in self.z_sample)

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur SUMO 
            ## et on stocke le résultat dans la liste SUMO sur laquelle on moyennera en sortie 
            SUMO.append(self.roulette_russe(I_0, Delta, K))

        return np.mean(SUMO)
    
    def log_likelihood_ML_RR(self, n_simulations):

        RR = []

        for _ in range(n_simulations):

            K = np.random.geometric(p=self.r, size=1)[0]

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample, z_sample_odd, z_sample_even = generate_encoder(self.x, K+3, noised_A=self.A, noised_b=self.B)

            ## erreur 
            ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
            Delta = lambda k: self.l_hat(2**(k+1), self.z_sample[:2**(k+1)+1]) - 1/2 * (self.l_hat(2**(k), z_sample_odd[:2**(k)+1]) + self.l_hat(2**(k), z_sample_even[:2**(k)+1]))

            I_0 = np.mean(self.l_hat(1, z) for z in self.z_sample)
            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur SUMO 
            ## et on stocke le résultat dans la liste SUMO sur laquelle on moyennera en sortie 
            RR.append(self.roulette_russe(I_0, Delta))

        return np.mean(RR)

    def log_likelihood_ML_SS(self):

        z_sample_odd = self.z_sample[1::2]
        z_sample_even = self.z_sample[::2]
        Delta = lambda k: self.l_hat(2**(k+1), self.z_sample[:2**(k+1)+1]) - 1/2 * (self.l_hat(2**(k), z_sample_odd[:2**(k)+1]) + self.l_hat(2**(k), z_sample_even[:2**(k)+1]))

        return self.single_sample(Delta)

    
