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
        
    AX_b = np.dot(x, noised_A.T) + noised_b

    cov = (2/3) * np.identity(dim) ## on calcule la variance de la normale multivariée associée à l'encodeur

    z_sample = np.random.multivariate_normal(AX_b, cov, size=k)
    z_odd = z_sample[1::2]
    z_even = z_sample[::2]

    ## POUR
    return z_sample , z_odd, z_even #AX_b #On return AX_b pour pouvoir les utiliser dans la fonction de décodage

def compute_ratio(x, z, theta, A, b):
    # Parameters
    I = np.eye(20)

    # Probability densities
    p_theta_xz = np.exp(-0.5 * np.dot(np.dot((z - theta).T, np.linalg.inv(2*I)), (z - theta)) - 0.5 * np.dot(np.dot((x - z).T, np.linalg.inv(I)), (x - z)))
    q_phi_z_given_x_density = np.exp(-0.5 * np.dot(np.dot((z - (np.dot(A, x) + b)).T, np.linalg.inv((2/3)*I)), (z - (np.dot(A, x) + b.flatten()))))

    # Compute the ratio
    ratio = p_theta_xz / q_phi_z_given_x_density

    return ratio

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

    def __init__(self, x, theta, r):
        self.x = x
        self.theta = theta
        self.r = r
        self.theta_hat = self.x.mean(axis=0)
        self.A_hat, self.b_hat = noised_params((1/2)*np.eye(20), (np.zeros(20) + self.theta_hat)/2)
        self.A, self.b = noised_params((1/2)*np.eye(20), (np.zeros(20) + self.theta)/2) ## on ajoute la valeur de A et b pour theta
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
        
        ## calcul des poids pour theta_hat (associé à l'observation x)
        weight_theta_hat = compute_ratio(self.x, z, self.theta_hat, self.A_hat, self.b_hat)

        ## cacul des poids pour theta (associé à un theta)                                                                                                    
        weight_theta = compute_ratio(self.x, z, self.theta, self.A, self.b)
        
        return (weight_theta_hat, weight_theta) ## attention, renvoie un tuple
       

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

        ## calcul de l_theta_hat (pour theta_hat)
        l_theta_hat = np.log((1/(len(z_sample)) * sum(self.weight(z_sample[i])[0] for i in range(len(z_sample)))))

        ## calcul de l_theta (pour theta)
        l_theta = np.log((1/(len(z_sample)) * sum(self.weight(z_sample[i])[1] for i in range(len(z_sample)))))

        return (l_theta_hat, l_theta)
    

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
    

    def single_sample(self, I_0, Delta, K):

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

        return I_0 + Delta(K)/(((1-self.r)**(K-1))*self.r)


    def log_likelihood_SUMO(self, n_simulations):

        SUMO_theta_hat = []
        SUMO_theta = []

        for _ in range(n_simulations):


            K = np.random.geometric(p=self.r, size=1)[0]

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample_hat, _, _ = generate_encoder(self.x, K+2, noised_A=self.A_hat, noised_b=self.b_hat)
            z_sample_theta, _, _ = generate_encoder(self.x, K+2, noised_A=self.A, noised_b=self.b)

            ## erreur 
            ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
            Delta_hat = lambda k: self.l_hat(z_sample_hat[:k+2])[0] - self.l_hat(z_sample_hat[:k+1])[0]  
            Delta_theta = lambda k: self.l_hat(z_sample_theta[:k+2])[1] - self.l_hat(z_sample_theta[:k+1])[1]

            I_0_hat = np.mean([self.l_hat(z)[0] for z in z_sample_hat])
            I_0_theta = np.mean([self.l_hat(z)[1] for z in z_sample_theta])

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur SUMO 
            ## et on stocke le résultat dans la liste SUMO sur laquelle on moyennera en sortie 
            SUMO_theta_hat.append(self.roulette_russe(I_0_hat, Delta_hat, K))
            SUMO_theta.append(self.roulette_russe(I_0_theta, Delta_theta, K))

        return (np.mean(SUMO_theta_hat), np.mean(SUMO_theta))
    

    def log_likelihood_ML_RR(self, n_simulations):

        RR_theta_hat = []
        RR_theta = []

        for _ in range(n_simulations):

            K = np.random.geometric(p=self.r, size=1)[0]

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample_hat, z_sample_odd_hat, z_sample_even_hat = generate_encoder(self.x, 2**(K+1), noised_A=self.A_hat, noised_b=self.b_hat)
            z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(self.x, 2**(K+1), noised_A=self.A, noised_b=self.b)


            ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
            Delta_hat = lambda k: self.l_hat(z_sample_hat[:2**(k+1)+1])[0] - 1/2 * (self.l_hat(z_sample_odd_hat[:2**(k)+1])[0] + self.l_hat(z_sample_even_hat[:2**(k)+1])[0])
            Delta_theta = lambda k: self.l_hat(z_sample_theta[:2**(k+1)+1])[1] - 1/2 * (self.l_hat(z_sample_odd_theta[:2**(k)+1])[1] + self.l_hat(z_sample_even_theta[:2**(k)+1])[1])


            I_0_hat = np.mean([self.l_hat(z)[0] for z in z_sample_hat])
            I_0_theta = np.mean([self.l_hat(z)[1] for z in z_sample_theta])

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
            ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
            RR_theta_hat.append(self.roulette_russe(I_0_hat, Delta_hat, K))
            RR_theta.append(self.roulette_russe(I_0_theta, Delta_theta, K))

        return (np.mean(RR_theta_hat), np.mean(RR_theta))


    def log_likelihood_ML_SS(self, n_simulations):

        SS_theta_hat = []
        SS_theta = []

        for _ in range(n_simulations):

            K = np.random.geometric(p=self.r, size=1)[0]

            ## K+3 pour avoir de quoi aller jusque K+3
            z_sample_hat, z_sample_odd_hat, z_sample_even_hat = generate_encoder(self.x, 2**(K+1), noised_A=self.A_hat, noised_b=self.b_hat)
            z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(self.x, 2**(K+1), noised_A=self.A, noised_b=self.b)


            ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
            Delta_hat = lambda k: self.l_hat(z_sample_hat[:2**(k+1)+1])[0] - 1/2 * (self.l_hat(z_sample_odd_hat[:2**(k)+1])[0] + self.l_hat(z_sample_even_hat[:2**(k)+1])[0])
            Delta_theta = lambda k: self.l_hat(z_sample_theta[:2**(k+1)+1])[1] - 1/2 * (self.l_hat(z_sample_odd_theta[:2**(k)+1])[1] + self.l_hat(z_sample_even_theta[:2**(k)+1])[1])


            I_0_hat = np.mean([self.l_hat(z)[0] for z in z_sample_hat])
            I_0_theta = np.mean([self.l_hat(z)[1] for z in z_sample_theta])

            ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
            ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
            SS_theta_hat.append(self.single_sample(I_0_hat, Delta_hat, K))
            SS_theta.append(self.single_sample(I_0_theta, Delta_theta, K))

        return (np.mean(SS_theta_hat), np.mean(SS_theta))


    def grad_SUMO(self, theta, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = theta - 5  # Limite inférieure de la plage
        theta_max = theta + 5 # Limite supérieure de la plage
        num_points = 20  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        SUMO_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, theta_values[i], self.r)
            SUMO_values.append(estimateur.log_likelihood_SUMO(n_simulations)[1])

        gradient_SUMO = np.gradient(SUMO_values, theta_values)

        return gradient_SUMO


    def grad_ML_RR(self, theta_star, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = theta_star - 5  # Limite inférieure de la plage
        theta_max = theta_star + 5 # Limite supérieure de la plage
        num_points = 30  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        ML_RR_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, theta_values[i], self.r)
            ML_RR_values.append(estimateur.log_likelihood_ML_RR(n_simulations)[1])
            print("Step: "+str('{:.1f}'.format(100*(i/30)))+"%")

        gradient_ML_RR = np.gradient(ML_RR_values, theta_values)

        return gradient_ML_RR
    

    def grad_ML_SS(self, theta, n_simulations):

        ## on se donne d'abord une plage de valeurs pour theta
        theta_min = theta - 5  # Limite inférieure de la plage
        theta_max = theta + 5 # Limite supérieure de la plage
        num_points = 30  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        ML_SS_values = []

        ## on caclue les valeurs de SUMO sur cette plage de valeurs
        for i in range(len(theta_values)):

            estimateur = Estimateurs(self.x, theta_values[i], self.r)
            ML_SS_values.append(estimateur.log_likelihood_ML_SS(n_simulations)[1])
            print("Step: "+str('{:.1f}'.format(100*(i/30)))+"%")

        gradient_ML_SS = np.gradient(ML_SS_values, theta_values)

        return gradient_ML_SS

    def true_likelihood(self, theta):

        likelihood = (1/((np.sqrt(2*np.pi)**20)*np.sqrt(2))) * np.exp(-(1/4)*np.sum((self.x - theta)**2))

        return np.log(likelihood)

    def true_grad(self, theta):

        return -0.5 * (self.x - theta*np.ones(20))
    

    def plot_likelihood(self, theta_true, n_simulations, methode='SUMO'):

        theta_min = theta_true - 5  # Limite inférieure de la plage
        theta_max = theta_true + 5 # Limite supérieure de la plage
        num_points = 30  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        if methode == 'SUMO':

            estimated_likelihood = []

            for theta in theta_values:
                estimateur = Estimateurs(self.x, theta, self.r)
                estimated_likelihood.append(estimateur.log_likelihood_SUMO(n_simulations)[1])

        elif methode == 'ML_RR':

            estimated_likelihood = []

            for theta in theta_values:
                estimateur = Estimateurs(self.x, theta, self.r)
                estimated_likelihood.append(estimateur.log_likelihood_ML_RR(n_simulations)[1])

        elif methode == 'ML_SS':

            estimated_likelihood = []

            for theta in theta_values:
                estimateur = Estimateurs(self.x, theta, self.r)
                estimated_likelihood.append(estimateur.log_likelihood_ML_SS(n_simulations)[1])

        true_likelihood_values = [self.true_likelihood(theta) for theta in theta_values]

        plt.plot(theta_values, true_likelihood_values, color='r', label='True likelihood')  
        plt.scatter(theta_values, estimated_likelihood, color='purple', marker='x', label=methode)
        plt.axvline(x=theta_true, color='black', linestyle='--', label='theta='+ str('{:.2f}'.format(theta_true)))
        plt.ylim([-300,500])
        plt.xlabel('Theta')
        plt.ylabel('Likelihood')
        plt.title(f'Estimation de la likelihood par {methode}')
        plt.legend(loc='best')
        plt.show()

        return true_likelihood_values, estimated_likelihood

'''    def plot_grad(self, theta_true, n_simulations, estimateur='SUMO'):

        theta_min = theta_true - 5  # Limite inférieure de la plage
        theta_max = theta_true + 5 # Limite supérieure de la plage
        num_points = 100  # Nombre de points à générer
        theta_values = np.linspace(theta_min, theta_max, num_points)

        if estimateur == 'SUMO':
            estimated_grad_values = self.grad_SUMO(theta, n_simulations)

        elif estimateur == 'ML_RR':
            estimated_grad_values = self.grad_ML_RR(theta, n_simulations)

        elif estimateur == 'ML_SS':
            estimated_grad_values = self.grad_ML_SS(theta_true, n_simulations)

        true_grad_values = [self.true_grad(theta) for theta in theta_values]'''

