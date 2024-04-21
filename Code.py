import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import time

### Pour l'installation automatique de tqdm

try:

    from tqdm import tqdm

except ImportError:

    print("tdqm package not found. Installing...")

    try:

        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
        
    except Exception as e:
        print(f"Error installing tqdm package: {e}")
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

def generate_encoder(x, k, noised_A, noised_b): ## on oublie l'idée generate_encoder(x_sample, A, b, dim=20) --> on a une expression explicite
                                        ## de A et b 
    """
    ---------------------------------------------------------------------------------------------------------------------
    IDÉE : Génère un échantillon i.i.d. z_sample de taille k selon la loi N(noised_Ax + noised_b, 2/3 * I) dans R^20, à 
           partir d'un échantillon i.i.d. x_sample.
    ---------------------------------------------------------------------------------------------------------------------
    """
    #A, b = noised_params((1/2)*np.eye(dim), (np.zeros(20) + theta_true)/2) ## on récupère les paramètres perturbés
    #Remarque : Dans l'article on ne tire pas avec theta_true mais avec theta_hat
        
    dim = 20

    AX_b = np.array(np.dot(noised_A, x) + noised_b).T

    cov = (2/3) * np.identity(dim) ## on calcule la variance de la normale multivariée associée à l'encodeur

    z_sample = []

    for _ in range(2**(k+1)):
        z_sample.append(np.random.multivariate_normal(AX_b, cov)) ## 2**(k+1) pour ne pas avoir de problèmes avec les échantillons pairs 
                                                                  ## et impairs
        
    z_odd = z_sample[1::2]
    z_even = z_sample[::2]

    ## POUR
    return z_sample , z_odd, z_even #AX_b #On return AX_b pour pouvoir les utiliser dans la fonction de décodage

def weights(x, z_sample, theta, A, b): #Question sur ce theta qui est censé être theta estimé sur X_data ? 

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
    
def true_likelihood(x, theta):

    likelihood = (1/((np.sqrt(2*np.pi)**20)*np.sqrt(2))) * np.exp(-(1/4)*np.sum((x - theta)**2))

    return np.log(likelihood)

def true_grad(x, theta):

    #return -0.5 * (x - theta*np.ones(20))
    return -0.5*(x.mean() - theta)

def log_likelihood_IWAE(theta, x, noised_A, noised_b, k, n_simulations):
        
        IWAE = []

        for _ in range(n_simulations):

            z_sample_theta, _, _ = generate_encoder(x, k, noised_A, noised_b) ## attention, la taille de l'échantillon est alors 2**(K+1)
                                                                              ## ce qui est plus grand que prévu, il faut slicer correctement

            weights_array = weights(x, z_sample_theta[:k], theta, noised_A, noised_b)
                
            l_hat_sum_k = (1/k)*np.sum(weights_array)

            IWAE_K = np.log(l_hat_sum_k)

            IWAE.append(IWAE_K)
    
        return np.mean(IWAE)

def log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations):
    
    SUMO = []

    # Initialize tqdm with the total number of simulations        
    for _ in range(n_simulations):

        K = np.random.geometric(p=r)

        ## K+3 pour avoir de quoi aller jusque K+3
        z_sample_theta, _, _ = generate_encoder(x, K+2, noised_A, noised_b) ## attention, la taille de l'échantillon est alors 2**(K+1)
                                                                            ## ce qui est plus grand que prévu, il faut slicer correctement

        weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)
            
        I_0 = np.mean([np.log(weights_array)])
    
        ## on se donne un delta particulier, celui qui correspond par définition à la méthode SUMO
        Delta_theta = lambda j: np.log(np.mean(weights_array[:j+2])) - np.log(np.mean(weights_array[:j+1]))

        SUMO_K = I_0
        
        for j in range(1, K):
            SUMO_K += Delta_theta(j)/((1-r)**j)

        SUMO.append(SUMO_K)

    return np.mean(SUMO)

def log_likelihood_ML_SS(r, theta, x, noised_A, noised_b, n_simulations):

    SS = []

    for _ in range(n_simulations):

        ## Étape 1 : on tire K ~ P(.) où P est la loi géométrique de paramètre 1
        K = np.random.geometric(p=r)

        ## Étape 2 : on tire notre échantillon ; ATTENTION, voir code de generate_encoder --> tire 2**(K+1) d'un coup
        z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(x, K, noised_A, noised_b)

        ## Étape 3 : on construit les vecteurs de poids
        weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)

        weights_array_odd = weights(x, z_sample_odd_theta, theta, noised_A, noised_b)
        weights_array_even = weights(x, z_sample_even_theta, theta, noised_A, noised_b)
        #weights_array_odd = np.log(z_sample_odd_theta) # impairs
        #weights_array_even = np.log(z_sample_even_theta) # pairs

        I_0 = np.mean([np.log(weights_array)])

        l_odd = np.log(np.mean(np.exp(np.log(weights_array_odd))))
        l_even = np.log(np.mean(np.exp(np.log(weights_array_even))))
        l_odd_and_even = np.log(np.mean(np.exp(np.log(weights_array))))

        ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
        Delta_theta_K = l_odd_and_even - 0.5 * (l_odd + l_even)

        ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
        ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
        SS.append(I_0 + (Delta_theta_K/(((1-r)**(K-1))*r)))



    return np.mean(SS)

def log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations):

    RR = []

    for _ in range(n_simulations):

        ## Étape 1 : on tire K ~ P(.) où P est la loi géométrique de paramètre 1
        K = np.random.geometric(p=r)

        ## Étape 2 : on tire notre échantillon ; ATTENTION, voir code de generate_encoder --> tire 2**(K+1) d'un coup
        z_sample_theta, z_sample_odd_theta, z_sample_even_theta = generate_encoder(x, K, noised_A, noised_b)

        ## Étape 3 : on construit les vecteurs de poids
        weights_array = weights(x, z_sample_theta, theta, noised_A, noised_b)

        weights_array_odd = weights(x, z_sample_odd_theta, theta, noised_A, noised_b)
        weights_array_even = weights(x, z_sample_even_theta, theta, noised_A, noised_b)
        #weights_array_odd = np.log(weights_array[1::2])
        #weights_array_even = np.log(weights_array[::2])

        I_0 = np.mean([np.log(weights_array)])

        l_odd = lambda j : np.log(np.mean(np.exp(np.log(weights_array_odd[:2**(j)]))))
        l_even = lambda j : np.log(np.mean(np.exp(np.log(weights_array_even[:2**(j)]))))
        l_odd_and_even = lambda j : np.log(np.mean(np.exp(np.log(weights_array[:2**(j+1)]))))

        ## on se donne un delta particulier, celui qui correspond par définition à la méthode RR
        Delta_theta = lambda j: l_odd_and_even(j) - 0.5 * (l_odd(j) + l_even(j))

        ## On clacule l'estimateur de la roulette russe associé à ce delta, c'est celui qui correspond à l'estimateur RR 
        ## et on stocke le résultat dans la liste RR sur laquelle on moyennera en sortie 
        #RR.append(I_0 + Delta_theta[0] + sum(Delta_theta(j)/((1-r)**(j-1)) for j in range(1, K+1)))
        RR.append(I_0 + Delta_theta(0) + sum(Delta_theta(j)/sum((1-r)**(i-1)*r for i in range(j, K+1)) for j in range(1, K+1)))

    return np.mean(RR)

def plot_likelihood(r, x, noised_A, noised_b, theta_true, n_simulations, k_IAWE, methode='SUMO'):

    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    progress_bar = tqdm(total=num_points, desc='Progression', position=0)

    if methode == 'SUMO':

        estimated_likelihood = []

        for theta in theta_values:

            estimated_likelihood.append(log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations))

            # Effectuer une tâche
            time.sleep(0.01)
            
            # Mettre à jour la barre de progression
            progress_bar.update(1)

        progress_bar.close()

    elif methode == 'ML_SS':

        estimated_likelihood = []

        for theta in theta_values:

            estimated_likelihood.append(log_likelihood_ML_SS(r, theta, x, noised_A, noised_b, n_simulations))

            # Effectuer une tâche
            time.sleep(0.01)
            
            # Mettre à jour la barre de progression
            progress_bar.update(1)

        progress_bar.close()

    elif methode == "ML_RR": 

        estimated_likelihood = []

        for theta in theta_values:

            estimated_likelihood.append(log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations))

            # Effectuer une tâche
            time.sleep(0.01)
            
            # Mettre à jour la barre de progression
            progress_bar.update(1)

        progress_bar.close()

    elif methode == "IWAE": 

        estimated_likelihood = []


        for theta in theta_values:

            estimated_likelihood.append(log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IAWE, n_simulations))
            
            # Effectuer une tâche
            time.sleep(0.01)
            
            # Mettre à jour la barre de progression
            progress_bar.update(1)

        progress_bar.close()

    elif methode == "all": 

        estimated_likelihood = [[], [], [], []]

        methodes = ['IWAE', 'ML_SS', 'ML_RR', 'SUMO']
        

        for theta in theta_values:
            
            estimated_likelihood[0].append(log_likelihood_ML_SS(r, theta, x, noised_A, noised_b, n_simulations))

            estimated_likelihood[1].append(log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations))

            estimated_likelihood[2].append(log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations))

            estimated_likelihood[3].append(log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IAWE, n_simulations))
            

            # Effectuer une tâche
            time.sleep(0.01)
            
            # Mettre à jour la barre de progression
            progress_bar.update(1)

        progress_bar.close()

        true_likelihood_values = [true_likelihood(x, theta) for theta in theta_values]

        # Création de la figure
        fig = go.Figure()

        # Ajout de la ligne pour la vraie vraisemblance
        fig.add_trace(go.Scatter(x=theta_values, y=true_likelihood_values, mode='lines', name='True likelihood', line=dict(color='red')))

        # Ajout des marqueurs pour les estimations de vraisemblance
        for i, estimated_likelihood_i in enumerate(estimated_likelihood):
            fig.add_trace(go.Scatter(x=theta_values, y=estimated_likelihood_i, mode='markers', name=methodes[i], marker=dict(color=['purple', 'orange', 'green', 'yellow'][i], symbol='x')))

        # Ajout de la ligne verticale pour theta_true
        fig.add_shape(type='line', x0=theta_true, x1=theta_true, y0=-300, y1=500, line=dict(color='black', dash='dash'), name=f'Theta={theta_true}')

        # Mise en forme de la figure
        fig.update_layout(
            xaxis=dict(title='Theta', tickfont=dict(size=12)),
            yaxis=dict(title='Likelihood', range=[-300, 500], tickfont=dict(size=12)),
            title='Estimation de la likelihood',
            legend=dict(font=dict(size=10)),
            showlegend=True,
            grid=dict(rows=1, columns=1)  # Définir le nombre de lignes et de colonnes de la grille
        )

        # Affichage de la figure
        fig.show()

        return
                
    true_likelihood_values = [true_likelihood(x, theta) for theta in theta_values]

    fig = go.Figure()

    # Tracé de la vraisemblance réelle
    fig.add_trace(go.Scatter(x=theta_values, y=true_likelihood_values, mode='lines', name='Vraisemblance réelle (avec theta_true)', line=dict(color='blue')))

    # Tracé de la vraisemblance estimée
    fig.add_trace(go.Scatter(x=theta_values, y=estimated_likelihood, mode='markers', name=methode, marker=dict(color='red', symbol='cross')))

    # Ligne verticale pour la valeur de theta vrai
    fig.add_shape(type='line', x0=theta_true, x1=theta_true, y0=-300, y1=500, line=dict(color='black', dash='dash'), name=f'Theta={theta_true}')

    # Mise en forme de la figure
    fig.update_layout(
        title=f'Estimation de la vraisemblance par {methode}',
        xaxis=dict(title='Theta', tickfont=dict(size=12)),
        yaxis=dict(title='Vraisemblance', range=[-300, 500], tickfont=dict(size=12)),
        legend=dict(font=dict(size=10)),
        showlegend=True,
        grid=dict(rows=1, columns=1)  # Définir le nombre de lignes et de colonnes de la grille
    )

    # Affichage de la figure
    fig.show()

    return 

def grad_IWAE(r, x, noised_A, noised_b, theta, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta - 5  # Limite inférieure de la plage
    theta_max = theta + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    IWAE_values = []

    ## on caclue les valeurs de SUMO sur cette plage de valeurs
    for i in range(len(theta_values)):

        IWAE_values.append(log_likelihood_IWAE(r, theta, x, noised_A, noised_b, n_simulations))

    gradient_IWAE = np.gradient(IWAE_values, theta_values)

    return gradient_IWAE

def grad_SUMO(r, x, noised_A, noised_b, theta, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta - 5  # Limite inférieure de la plage
    theta_max = theta + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    SUMO_values = []

    ## on caclue les valeurs de SUMO sur cette plage de valeurs
    for i in range(len(theta_values)):

        SUMO_values.append(log_likelihood_SUMO(r, x, noised_A, noised_b, theta, n_simulations))

    gradient_SUMO = np.gradient(SUMO_values, theta_values)

    return gradient_SUMO

def grad_ML_RR(r, x, noised_A, noised_b, theta, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta - 5  # Limite inférieure de la plage
    theta_max = theta + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ML_RR_values = []

    ## on caclue les valeurs de SUMO sur cette plage de valeurs
    for i in range(len(theta_values)):

        ML_RR_values.append(log_likelihood_ML_RR(r, x, noised_A, noised_b, theta, n_simulations))
        #print("Step: "+str('{:.1f}'.format(100*(i/30)))+"%")

    gradient_ML_RR = np.gradient(ML_RR_values, theta_values)

    return gradient_ML_RR
    
def grad_ML_SS(r, x, noised_A, noised_b, theta, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta - 5  # Limite inférieure de la plage
    theta_max = theta + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ML_SS_values = []

    ## on caclue les valeurs de SUMO sur cette plage de valeurs
    for i in range(len(theta_values)):

        ML_SS_values.append(log_likelihood_ML_SS(r, x, noised_A, noised_b, theta, n_simulations))
        #print("Step: "+str('{:.1f}'.format(100*(i/30)))+"%")

    gradient_ML_SS = np.gradient(ML_SS_values, theta_values)

    return gradient_ML_SS

def plot_gradient(r, x, noised_A, noised_b, theta_true, n_simulations, methode='SUMO'):
    
    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    num_points = 60  # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    if methode == 'SUMO':

        estimated_grad = grad_SUMO(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'ML_SS':

        estimated_grad = grad_ML_SS(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'ML_RR': 

        estimated_grad = grad_ML_RR(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'IWAE': 

        estimated_grad = grad_IWAE(r, x, noised_A, noised_b, theta_true, n_simulations)

    #elif methode == 'all': 
                
    true_gradient_values = [true_grad(x, theta) for theta in theta_values]

    # Créer la figure
    fig = go.Figure()

    # Ajouter la vraie valeur du gradient
    fig.add_trace(go.Scatter(x=theta_values, y=true_gradient_values, mode='lines', name='True Gradient', line=dict(color='red')))

    # Ajouter l'estimation du gradient
    fig.add_trace(go.Scatter(x=theta_values, y=estimated_grad, mode='markers', name=methode, marker=dict(color='purple', symbol='x')))

    # Ajouter la ligne verticale pour la vraie valeur de theta
    fig.add_shape(type='line', x0=theta_true, x1=theta_true, y0=min(min(true_gradient_values), min(estimated_grad)), y1=max(max(true_gradient_values), max(estimated_grad)), 
                line=dict(color='black', width=2, dash='dash'), name=f'theta={theta_true}')

    # Mise en forme de la figure
    fig.update_layout(
        xaxis=dict(title='Theta'),
        yaxis=dict(title='Gradient'),
        title=f'Estimation du gradient par {methode}',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)),
        showlegend=True
    )

    # Affichage de la figure
    fig.show()