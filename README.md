
<h1 align="center"> Projet de Simulation and Monte Carlo Methods à l'ENSAE Paris </h1> <br>
<p align="center">
  <a href="https://github.com/NailKhelifa/MLMC_Unibaised_Gradient_Estimation_for_Deep_LVM">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/LOGO-ENSAE.png" alt="Logo" width="400" height="400">
  </a>
</p>

<p align="center">
  <strong>On Multilevel Monte Carlo Unbiased Gradient Estimation For Deep Latent Variable Models.</strong>
</p>

<p align="center">
  Tom Rossa, Axel Pinçon et Naïl Khelifa
  <br />
  <br />
  
</p>



<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Introduction - Context](#introduction)
- [Structure du projet](#structure)
- [Authors](#authors)
- [Licence](#license)
- [Acknowledgments](#acknowledgments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction - Context

In Markowitz's Mean Variance Theory, investors focus on the mean and variance of portfolio returns. Portfolio optimization, known as **Mean-Variance Optimization (MVO)**, seeks to maximize returns while limiting variance. The optimization problem is expressed as:

```math
\min ~ \frac{1}{2}<\mathbf{\omega}, \Sigma \mathbf{\omega}>
\text{ subject to }
\quad \rho(\mathbf{\omega}) = \rho_0
\quad \sum_{i=1}^N \omega_i = 1
```

Here, $\Sigma$ is the covariance matrix, $\rho(\mathbf{\omega})$ is the reward for allocation $\mathbf{\omega}$, and $\rho_0$ is a fixed reward level.

For MVO, estimating returns and the covariance matrix $\Sigma$ is crucial. However, in a large asset universe (e.g., 700 stocks), inverting $\Sigma$ can be problematic due to estimation complexities. Estimating $\hat{\Sigma}$ may result in a non-invertible matrix, and even if $\Sigma$ is invertible, it is often ill-conditioned in a large asset universe.

To address this, traditional approaches modify $\hat{\Sigma}$ numerically or shrink it towards an invertible target matrix. However, these methods have drawbacks. Instead, we tackle these challenges using a different approach: clustering.

Our aim is to identify groups of similar returns through clustering, grouping assets for better diversification. We divide the assets into fewer subgroups (clusters) and allocate weights uniformly within each group based on clustering algorithms using return and fundamental data. Finally, classical MVO is performed on the new asset groups.

## Structure


## Authors

  - Tom Rossa
    
  - Axel Pinçon
    
  - Naïl Khelifa


## License

Distributed under the [MIT License](LICENSE.md). See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
