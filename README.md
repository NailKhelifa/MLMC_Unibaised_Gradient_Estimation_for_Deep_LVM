
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

A very popular technique in Machine Learning is to maximise the likelihood of the considered model through SGD (stochastic gradient descent), which is gradient descent (applied to minus the likelihood) with the true gradient replaced by an unbiased estimate. The following paper presents a novel method to obtain unbiased estimates of gradient for latent variable models: http://proceedings.mlr.press/v130/shi21d.html. The objective of this project is to reproduce their first numerical experiment, and compare their two estimators (based on either the simple term estimate, or the Russian roulette one) with IWAE (which is biased).

## Structure


## Authors

  - Tom Rossa
    
  - Axel Pinçon
    
  - Naïl Khelifa


## License

Distributed under the [MIT License](LICENSE.md). See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
