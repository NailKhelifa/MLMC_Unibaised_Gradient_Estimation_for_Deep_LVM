import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys

### Pour l'installation automatique de tqdm

try:

    from tqdm import tqdm

except ImportError:

    print("tdqm package not found. Installing...")

    try:

        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from pypfopt.efficient_frontier import EfficientFrontier
        
    except Exception as e:
        print(f"Error installing PyPortfolioOpt package: {e}")
        sys.exit(1)

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
    z = np.random.multivariate_normal(np.zeros(dim) + theta*np.ones(20), np.identity(dim))

    # Génère l'échantillon X = (X_1,...,X_20) suivant la loi conditionnelle à Z :  N(x | z, Id)
    x = np.random.multivariate_normal(z, np.eye(dim))

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
    
    return np.array(noised_A), np.array(noised_b)

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

    AX_b = np.array(np.dot(noised_A, x) + noised_b).T

    cov = (2/3) * np.identity(dim) ## on calcule la variance de la normale multivariée associée à l'encodeur

    z_sample = []

    for _ in range(k):
        z_sample.append(np.random.multivariate_normal(AX_b, cov)) ## 2**(k+1) pour ne pas avoir de problèmes avec les échantillons pairs et impairs

    ## POUR
    return z_sample #AX_b #On return AX_b pour pouvoir les utiliser dans la fonction de décodage
   
def true_likelihood(x, theta):

    likelihood = (1/((np.sqrt(2*np.pi)**20)*np.sqrt(2))) * np.exp(-(1/4)*np.sum((x - theta)**2))

    return np.log(likelihood)

def true_grad(x, theta):

    return -0.5 * (x - theta*np.ones(20))

class Estimateurs: 

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

    def __init__(self, x, z_sample, theta, r):
        self.x = x
        self.z_sample = z_sample
        self.z_sample_odd = self.z_sample[1::2]
        self.z_sample_even = self.z_sample[::2]
        self.theta = theta
        self.dim = 20
        self.r = r
        self.A, self.b = noised_params((1/2)*np.eye(20), (np.zeros(20) + self.theta)/2) ## on ajoute la valeur de A et b pour theta
        self.I_0 = np.mean(self.l_hat(1, z) for z in self.z_sample)


    def weights(self, z):
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
        
        # Parameters
        AX_b = np.dot(self.A, self.x) + self.b
        I = np.eye(self.dim)
        theta_mean = self.theta * np.ones(self.dim)
        weights = []
    
        for z_ in z:

            # Probability densities
            q_phi_z_given_x = multivariate_normal.pdf(z_, mean=AX_b, cov=(2/3)*I)
            p_theta_xz = multivariate_normal.pdf(z_, mean=theta_mean, cov=I)*multivariate_normal.pdf(self.x, mean=z_, cov=I)

            weights.append(p_theta_xz / q_phi_z_given_x)

        return weights
       

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

        return np.log((1/(len(z_sample)) * sum(self.weights(self.z_sample))))
    

    def roulette_russe(self, Delta, K):

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

        return self.I_0 + sum([Delta(0)] + [Delta(k)/((1-self.r)**(k-1)) for k in range(1,K + 1)])
    

    def single_sample(self, Delta, K):

        '''
        ---------------------------------------------------------------------------------------------------------------------
        IDEA: compute the single_sample estimator (ss) under the conditions of theorem 1 and a geometric distribution of 
              parameter r for the number of parameters in the sum 
        ---------------------------------------------------------------------------------------------------------------------

        ---------------------------------------------------------------------------------------------------------------------
        ARGUMENTS: 
        - I_0: float ; see the theoretical framework in the article for more details
        
        - Delta: lambda function to compute the ∆_k such as in the theoretical framework of the paper

        - K: integer, sampled according to a law on N (positive integers)
        ---------------------------------------------------------------------------------------------------------------------
        '''

        return self.I_0 + Delta(K)/(((1-self.r)**(K-1))*self.r)


    def log_likelihood_SUMO(self, n_simulations):

        SUMO = []

        # Initialize tqdm with the total number of simulations
        with tqdm(total=n_simulations) as pbar:
            
            for _ in range(n_simulations):

                K = np.random.geometric(p=self.r)

                ## K+3 pour avoir de quoi aller jusque K+3
                self.z_sample = generate_encoder(self.x, K+3, self.A, self.b) ## attention, la taille de l'échantillon est alors 2**(K+1)
                                                                                ## ce qui est plus grand que prévu, il faut slicer correctement
            
                ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
                Delta = lambda j: np.log(np.mean(self.weights(self.z_sample)[:j+2])) - np.log(np.mean(self.weights(self.z_sample)[:j+1]))

                SUMO.append(self.roulette_russe(Delta, K))

                pbar.update(1)  # Update the progress bar

        return np.mean(SUMO)
    

    def log_likelihood_ML_RR(self, n_simulations):

        RR = []

        with tqdm(total=n_simulations) as pbar:

            for _ in range(n_simulations):

                ## Étape 1 : on tire K ~ P(.) où P est la loi géométrique de paramètre 1
                K = np.random.geometric(p=self.r)

                ## Étape 2 : on tire notre échantillon ; ATTENTION, voir code de generate_encoder --> tire 2*(K+1) d'un coup
                self.z_sample = generate_encoder(self.x, 2**(K+1), self.A, self.b)

                Delta = lambda j: np.log(np.mean(self.weights(self.z_sample)[:2**(j+1)])) - 0.5 * (np.log(np.mean(self.weights(self.z_sample_odd)[:2**j])) + np.log(np.mean(self.weights(self.z_sample_even)[:2**j])))

                ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
                ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
                RR.append(self.roulette_russe(Delta, K))

                pbar.update(1)

        return np.mean(RR)


    def log_likelihood_ML_SS(self, n_simulations):

        SS = []

        with tqdm(total=n_simulations) as pbar:

            for _ in range(n_simulations):

                ## Étape 1 : on tire K ~ P(.) où P est la loi géométrique de paramètre 1
                K = np.random.geometric(p=self.r)

                ## Étape 2 : on tire notre échantillon ; ATTENTION, voir code de generate_encoder --> tire 2*(K+1) d'un coup
                self.z_sample = generate_encoder(self.x, 2**(K+1), self.A, self.b)

                Delta = lambda j: np.log(np.mean(self.weights(self.z_sample)[:2**(j+1)])) - 0.5 * (np.log(np.mean(self.weights(self.z_sample_odd)[:2**j])) + np.log(np.mean(self.weights(self.z_sample_even)[:2**j])))

                ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
                ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
                SS.append(self.single_sample(Delta, K))

                pbar.update(1)

        return np.mean(SS)


    def grad_SUMO(self, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = self.theta - 5  # Limite inférieure de la plage
        theta_max = self.theta + 5 # Limite supérieure de la plage
        num_points = 20  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        SUMO_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, self.z_sample, theta_values[i], self.r)
            SUMO_values.append(estimateur.log_likelihood_SUMO(n_simulations))

        gradient_SUMO = np.gradient(SUMO_values, theta_values)

        return gradient_SUMO


    def grad_ML_RR(self, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = self.theta - 5  # Limite inférieure de la plage
        theta_max = self.theta + 5 # Limite supérieure de la plage
        num_points = 30  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        ML_RR_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, self.z_sample, theta_values[i], self.r)
            ML_RR_values.append(estimateur.log_likelihood_ML_RR(n_simulations))

        gradient_ML_RR = np.gradient(ML_RR_values, theta_values)

        return gradient_ML_RR
    

    def grad_ML_SS(self, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = self.theta - 5  # Limite inférieure de la plage
        theta_max = self.theta + 5 # Limite supérieure de la plage
        num_points = 30  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        ML_SS_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, self.z_sample, theta_values[i], self.r)
            ML_SS_values.append(estimateur.log_likelihood_ML_SS(n_simulations))

        gradient_ML_SS = np.gradient(ML_SS_values, theta_values)

        return gradient_ML_SS



def plot_likelihood(x, z, theta_true, r, n_simulations, methode='SUMO'):

    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    num_points = 30  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    if methode == 'SUMO':

        estimated_likelihood = []

        for theta in theta_values:
            estimateur = Estimateurs(x, z, theta, r)
            estimated_likelihood.append(estimateur.log_likelihood_SUMO(n_simulations))


    elif methode == 'ML_SS':

        for theta in theta_values:
            estimateur = Estimateurs(x, z, theta, r)
            estimated_likelihood.append(estimateur.log_likelihood_ML_RR(n_simulations))
    
    elif methode == 'ML_SS':

        for theta in theta_values:
            estimateur = Estimateurs(x, z, theta, r)
            estimated_likelihood.append(estimateur.log_likelihood_ML_SS(n_simulations))
                
    true_likelihood_values = [true_likelihood(x, theta) for theta in theta_values]

    plt.plot(theta_values, true_likelihood_values, color='r', label='True likelihood')  
    plt.scatter(theta_values, estimated_likelihood, color='purple', marker='x', label=methode)
    plt.axvline(x=theta_true, color='black', linestyle='--', label='theta='+ str('{:.2f}'.format(theta_true)))
    plt.ylim([-300,500])
    plt.xlabel('Theta')
    plt.ylabel('Likelihood')
    plt.title(f'Estimation de la likelihood par {methode}')
    plt.legend(loc='best')
    plt.show()

    return None