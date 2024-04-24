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

#####################################################################################################################################
###################################################### FONCTIONS FONDAMENTALES ######################################################
#####################################################################################################################################

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

    return 10*(x.mean() - theta)

#####################################################################################################################################
################################################## ESTIMATION DE LA VRAISEMBLANCE ###################################################
#####################################################################################################################################


def log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IWAE, n_simulations):
        
        IWAE = []

        for _ in range(n_simulations):

            z_sample_theta, _, _ = generate_encoder(x, int(np.log(k_IWAE)/np.log(2)), noised_A, noised_b) ## attention, quand k_IWAE = 20 on en tire 2**21
                                                                                ## Or, on en veut que k_IWAE

            weights_array = weights(x, z_sample_theta[:k_IWAE], theta, noised_A, noised_b)
                
            l_hat_sum_k = (1/k_IWAE)*np.sum(weights_array)

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

def plot_likelihood(r, x, noised_A, noised_b, theta_true, n_simulations, k_IWAE = 5, methode='SUMO'):

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

            estimated_likelihood.append(log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IWAE, n_simulations))
            
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

            estimated_likelihood[3].append(log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IWAE, n_simulations))
            

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

def plot_errors_likelihood(r, theta_true, x, noised_A, noised_b, n_simulations, n_runs, methode, k_IWAE = 5):
    #On fixe k_IWAE = 5 dans l'argument pour éviter de le passer en argument à chaque fois
    # Définition des valeurs initiales
    theta_min = theta_true - 5
    theta_max = theta_true + 5
    num_points = 30
    theta_values = np.linspace(theta_min, theta_max, num_points)
    n_runs = 5  # Nombre de lancements du programme
    true_likelihood_values = [true_likelihood(x, theta) for theta in theta_values]

    estimated_likelihood = []
    
    if methode == 'SUMO':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            likelihoods = []
            for _ in range(n_runs):
                likelihoods.append(log_likelihood_SUMO(r, theta, x, noised_A, noised_b, n_simulations))
            estimated_likelihood.append(likelihoods)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'IWAE':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            likelihoods = []
            for _ in range(n_runs):
                likelihoods.append(log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IWAE, n_simulations))
            estimated_likelihood.append(likelihoods)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'ML_SS':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            likelihoods = []
            for _ in range(n_runs):
                likelihoods.append(log_likelihood_ML_SS(r, theta, x, noised_A, noised_b, n_simulations))
            estimated_likelihood.append(likelihoods)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'ML_RR':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            likelihoods = []
            for _ in range(n_runs):
                likelihoods.append(log_likelihood_ML_RR(r, theta, x, noised_A, noised_b, n_simulations))
            estimated_likelihood.append(likelihoods)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    # Création de la figure
    plt.figure(figsize=(15, 8))

    # Tracé de la vraisemblance réelle
    plt.plot(theta_values, true_likelihood_values, color='blue', label='Vraisemblance réelle')

    # Tracé de la vraisemblance estimée
    plt.scatter(theta_values, [np.median(x) for x in estimated_likelihood], color='red', label='Estimation moyenne')

    # Ajout des boxplots
    plt.boxplot(estimated_likelihood, positions=theta_values, widths=0.1, showfliers=False, patch_artist=True, notch=False, showmeans=True, boxprops=dict(facecolor='orange'))

    # Ligne verticale pour la valeur de theta_true vrai
    plt.axvline(x=theta_true, color='black', linestyle='--', label=f'Theta={theta_true}')

    # Mise en forme de la figure
    plt.title(f'Estimation de la vraisemblance par {methode} avec incertitude pour {n_runs} répétitions')
    plt.xlabel('Theta')
    plt.ylabel('Vraisemblance')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(min(theta_values), max(theta_values)+1, 2))


    # Affichage de la figure
    plt.show()

    return None

#####################################################################################################################################
###################################################### ESTIMATION DU GRADIENT #######################################################
#####################################################################################################################################


def grad_IWAE(x, noised_A, noised_b, theta, k_IWAE, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta - 5  # Limite inférieure de la plage
    theta_max = theta + 5 # Limite supérieure de la plage
    step = 0.2
    num_points = int((theta_max - theta_min) / step) # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ## on caclue les valeurs de IWAE sur cette plage de valeurs
    IWAE_values = [log_likelihood_IWAE(theta, x, noised_A, noised_b, k_IWAE, n_simulations) for theta in theta_values]

    gradient_IWAE = np.gradient(IWAE_values)

    return gradient_IWAE

def grad_SUMO(r, x, noised_A, noised_b, theta_true, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    step = 0.2
    num_points = int((theta_max - theta_min) / step) # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ## on caclue les valeurs de SUMO sur cette plage de valeurs
    SUMO_values = []

    for theta in theta_values:
        SUMO_values.append(log_likelihood_SUMO(r, x, noised_A, noised_b, theta, n_simulations))

    gradient_SUMO = np.gradient(SUMO_values)

    return gradient_SUMO

def grad_ML_RR(r, x, noised_A, noised_b, theta_true, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    step = 0.2
    num_points = int((theta_max - theta_min) / step)
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ## on caclue les valeurs de ML_RR sur cette plage de valeurs
    ML_RR_values = [log_likelihood_ML_RR(r, x, noised_A, noised_b, theta, n_simulations) for theta in theta_values]

    gradient_ML_RR = np.gradient(ML_RR_values)

    return gradient_ML_RR
    
def grad_ML_SS(r, x, noised_A, noised_b, theta_true, n_simulations):

    ## on se donne d'abord une plage de valeurs pour theta
    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    step = 0.2
    num_points = int((theta_max - theta_min) / step) # Nombre de points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)

    ## on caclue les valeurs de ML_SS sur cette plage de valeurs
    ML_SS_values = [log_likelihood_ML_SS(r, x, noised_A, noised_b, theta, n_simulations) for theta in theta_values]

    gradient_ML_SS = np.gradient(ML_SS_values)

    return gradient_ML_SS

def plot_gradient(r, x, noised_A, noised_b, theta_true, n_simulations, methode, k_IWAE = 5):
    # On fixe k_IWAE pour éviter de le passer en argument à chaque fois + methode non fixée pour éviter d'afficher 'SUMO' dans le titre

    if methode == 'SUMO':

        estimated_grad = grad_SUMO(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'ML_SS':

        estimated_grad = grad_ML_SS(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'ML_RR': 

        estimated_grad = grad_ML_RR(r, x, noised_A, noised_b, theta_true, n_simulations)

    elif methode == 'IWAE': 

        estimated_grad = grad_IWAE(r, x, noised_A, noised_b, theta_true, k_IWAE, n_simulations)

    #elif methode == 'all': 
        
    # On récupère la plage de valeur sur laquelle on évalue
    theta_min = theta_true - 5  # Limite inférieure de la plage
    theta_max = theta_true + 5 # Limite supérieure de la plage
    step = 0.2
    num_points = int((theta_max - theta_min) / step)  # 50 points à générer
    theta_values = np.linspace(theta_min, theta_max, num_points)
                
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

def plot_errors_gradient(r, theta_true, x, noised_A, noised_b, n_simulations, n_runs, k_IWAE = 5, methode='SUMO'):
    #On fixe k_IWAE = 5 dans l'argument pour éviter de le passer en argument à chaque fois
    # Définition des valeurs initiales
    theta_min = theta_true - 5
    theta_max = theta_true + 5
    step = 0.2
    num_points = int((theta_max - theta_min) / step)
    theta_values = np.linspace(theta_min, theta_max, num_points)

    true_gradient_values = [true_grad(x, theta) for theta in theta_values]

    estimated_grad = []
    
    if methode == 'SUMO':
        # Calcul des gradients estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            grad = []
            for _ in range(n_runs):
                grad.append(grad_SUMO(r, x, noised_A, noised_b, theta, n_simulations))
            estimated_grad.append(grad)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'IWAE':
        # Calcul des gradients estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            grad = []
            for _ in range(n_runs):
                grad.append(grad_IWAE(x, noised_A, noised_b, theta, k_IWAE, n_simulations))
            estimated_grad.append(grad)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'ML_SS':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            grad = []
            for _ in range(n_runs):
                grad.append(grad_ML_SS(r, x, noised_A, noised_b, theta, n_simulations))
            estimated_grad.append(grad)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    if methode == 'ML_RR':
        # Calcul des vraisemblances estimées et stockage des résultats pour chaque exécution
        progress_bar = tqdm(total=len(theta_values), desc='Progression', position=0)
        for theta in theta_values:
            grad = []
            for _ in range(n_runs):
                grad.append(grad_ML_RR(r, x, noised_A, noised_b, theta, n_simulations))
            estimated_grad.append(grad)
            time.sleep(0.01)
            progress_bar.update(1)
        progress_bar.close()

    # Création de la figure
    plt.figure(figsize=(15, 8))

    # Tracé de la vraisemblance réelle
    plt.plot(theta_values, true_gradient_values, color='blue', label='Gradient réel')

    # Tracé de la vraisemblance estimée
    plt.scatter(theta_values, [np.median(x) for x in estimated_grad], color='red', label='Estimation moyenne')

    # Ajout des boxplots
    plt.boxplot(estimated_grad, positions=theta_values, widths=0.1, showfliers=False, patch_artist=True, notch=False, showmeans=True, boxprops=dict(facecolor='orange'))

    # Ligne verticale pour la valeur de theta_true vrai
    plt.axvline(x=theta_true, color='black', linestyle='--', label=f'Theta={theta_true}')

    # Mise en forme de la figure
    plt.title(f'Estimation du gradient par {methode} avec incertitude pour {n_runs} répétitions')
    plt.xlabel('Theta')
    plt.ylabel('Gradient')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(min(theta_values), max(theta_values)+1, 2))


    # Affichage de la figure
    plt.show()

    return None


#####################################################################################################################################
###################################################### ANLAYSE BIAIS-VARIANCE #######################################################
#####################################################################################################################################

def plot_bias_likelihood(x, theta_true, noised_A, noised_b, n_simulations, k_IWAE = 5): 

    # Définition des valeurs initiales
    cost_min = 6
    cost_max = 384
    step = 6
    cost_values = np.arange(cost_min, cost_max, step)

    #Calcul de la vraisemblance vraie
    param_true = true_likelihood(x, theta_true)

    bias = [], [], [], []

    for i in range(len(cost_values)):

            bias[0].append((log_likelihood_SUMO(1/cost_values[i], theta_true, x, noised_A, noised_b, n_simulations) - param_true)^2)

            bias[1].append((log_likelihood_ML_SS(1/cost_values[i], theta_true, x, noised_A, noised_b, n_simulations) - param_true)^2)

            bias[2].append((log_likelihood_ML_RR(1/cost_values[i], theta_true, x, noised_A, noised_b, n_simulations) - param_true)^2) 

            bias[3].append((log_likelihood_IWAE(theta_true, x, noised_A, noised_b, k_IWAE, n_simulations) - param_true)^2)

        #print("Pour r = "+str(cost_values[i])+" le biais est de "+str(bias))
    
    return bias



#####################################################################################################################################
###################################################### PARTIE DU CODE POUR SGD ######################################################
#####################################################################################################################################

