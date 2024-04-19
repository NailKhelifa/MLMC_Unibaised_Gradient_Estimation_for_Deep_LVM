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
        z_sample.append(np.random.multivariate_normal(AX_b, cov)) ## 2**(k+1) pour ne pas avoir de problèmes avec les échantillons pairs 
                                                                  ## et impairs
        
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
    
def true_likelihood(x, theta):

    likelihood = (1/((np.sqrt(2*np.pi)**20)*np.sqrt(2))) * np.exp(-(1/4)*np.sum((x - theta)**2))

    return np.log(likelihood)

def true_grad(x, theta):

    return -0.5 * (x - theta*np.ones(20))

def log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations):
    
    SUMO_theta = []

    # Initialize tqdm with the total number of simulations
    with tqdm(total=n_simulations) as pbar:
        
        for _ in range(n_simulations):

            K = np.random.geometric(p=r)

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample_theta, _, _ = generate_encoder(x, K+2, noised_A, noised_b) ## attention, la taille de l'échantillon est alors 2**(K+1)
                                                                              ## ce qui est plus grand que prévu, il faut slicer correctement

            weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)
                
            I_0 = np.mean([np.log(weights_array)])
        
            ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
            Delta_theta = lambda j: np.log(np.mean(weights_array[:j+2])) - np.log(np.mean(weights_array[:j+1]))

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur SUMO 
            ## et on stocke le résultat dans la liste SUMO sur laquelle on moyennera en sortie 
            SUMO_theta.append(roulette_russe(r, I_0, Delta_theta, K))

            pbar.update(1)  # Update the progress bar

    return np.mean(SUMO_theta)


def log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations):

    RR = []

    with tqdm(total=n_simulations) as pbar:

        for _ in range(n_simulations):

            K = np.random.geometric(p=r)

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(x, 2**(K+1), noised_A, noised_b)

            weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)

            weights_array_odd = np.log(weights_array[1::2])
            weights_array_even = np.log(weights_array[::2])

            I_0 = np.mean([np.log(weights_array)])

            ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
            Delta_theta = lambda j: np.log(np.mean(weights_array)) - 0.5 * (np.log)

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
            ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
            RR.append(roulette_russe(I_0, Delta_theta, K))

    return np.mean(RR)


def plot_likelihood(r, x, noised_A, noised_b, theta_true, n_simulations, methode='SUMO'):

    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    num_points = 30  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    if methode == 'SUMO':

        estimated_likelihood = []
        with tqdm(total=n_simulations) as pbar:

            for theta in theta_values:

                estimated_likelihood.append(log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations))

                pbar.update(1)

    #elif methode == 'ML_RR':

    #   estimated_likelihood = []

        #for theta in theta_values:
        #    estimateur = Estimateurs(self.x, theta, self.r)
        #    estimated_likelihood.append(estimateur.log_likelihood_ML_RR(n_simulations)[1])'''

    #elif methode == 'ML_SS':
    #    estimated_likelihood = []

        #for theta in theta_values:
        #    estimateur = Estimateurs(self.x, theta, self.r)
        #    estimated_likelihood.append(estimateur.log_likelihood_ML_SS(n_simulations)[1])

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

    return 