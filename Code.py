import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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
        
    AX_b = np.dot(noised_A, x) + noised_b

    cov = (2/3) * np.identity(dim) ## on calcule la variance de la normale multivariée associée à l'encodeur

    z_sample = np.random.multivariate_normal(AX_b, cov, size=k)
    z_odd = z_sample[1::2]
    z_even = z_sample[::2]

    ## POUR
    return z_sample , z_odd, z_even #AX_b #On return AX_b pour pouvoir les utiliser dans la fonction de décodage

def weights(x, z_sample, theta, A, b):

    dimension = 20

    # Parameters
    AX_b = np.dot(A, x) + b
    I = np.eye(dimension)
    theta_mean = theta*np.ones(dimension)
    weights = []
    
    for z in z_sample:

        # Probability densities
        q_phi_z_given_x_density = multivariate_normal.pdf(z, mean=AX_b, cov=(2/3)*I)
        p_theta_xz = multivariate_normal.pdf(z, mean=theta_mean, cov=I)*multivariate_normal.pdf(x, mean=z, cov=I)

        weights.append(p_theta_xz / q_phi_z_given_x_density)

    return weights


def roulette_russe(r, I_0, Delta, K):

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

    return I_0 + sum(Delta(k)/((1-r)**(k)) for k in range(1,K))

def single_sample(r, I_0, Delta, K):

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

    return I_0 + Delta(K)/(((1-r)**(K-1))*r)
    
def log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations):
    
    SUMO_theta = []

    for _ in range(n_simulations):

        K = np.random.geometric(p=r)

        ## K+3 pour avoir de quoi aller jusque K+3
        z_sample_theta, _, _ = generate_encoder(x, K+2, noised_A, noised_b)

        weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)
            
        I_0 = np.mean([np.log(weights_array)])
    
        ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
        Delta_theta = lambda j: np.log(np.mean(weights_array[:j+2])) - np.log(np.mean(weights_array[:j+1]))

        ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur SUMO 
        ## et on stocke le résultat dans la liste SUMO sur laquelle on moyennera en sortie 
        SUMO_theta.append(roulette_russe(I_0, Delta_theta, K))

    return np.mean(SUMO_theta)

def log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations):

    RR = []

    for _ in range(n_simulations):

        K = np.random.geometric(p=r)

        ## K+3 pour avoir de quoi aller jusque K+3
        z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(x, 2**(K+1), noised_A, noised_b)

        weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)
        weights_array_odd = weights_array[1::2]
        weights_array_even = weights_array[::2]

        I_0 = np.mean([np.log(weights_array)])

        ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
        Delta_theta = lambda k: l_hat(z_sample_theta[:2**(k+1)+1])[1] - 1/2 * (self.l_hat(z_sample_odd_theta[:2**(k)+1])[1] + self.l_hat(z_sample_even_theta[:2**(k)+1])[1])

        ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
        ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
        RR.append(roulette_russe(I_0, Delta_theta, K))

    return np.mean(RR)