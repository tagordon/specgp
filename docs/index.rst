.. SpecGP documentation master file, created by
   sphinx-quickstart on Mon Jul 13 21:52:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpecGP
==================================
*SpecGP* is an extension for *exoplanet* and 
*celerite* that enables 2D Gaussian process 
models. While there are a myriad of uses 
for a 2D GP, one use of special relevance 
to exoplanet astronomy is to allow for simulteaneous 
modeling of stellar variability at multiple 
wavelengths. 

*SpecGP* provides a new *celerite* term, ``KroneckerTerm``, 
which combines a 1D *celerite* term with a 
second covariance matrix for the GP's 
second dimension. In the case of multiwavelength 
stellar variability, this second covariance is the 
outer product of a vector which we call ``alpha`` 
with itself. In this case ``alpha`` is a vector 
of scale factors representing the relative amplitudes 
of the variability in each wavelength. 

*SpecGP* is fast and scalable, running in order 
N time for a grid of N points in the multiwavelength 
case. As mentioned above, *SpecGP* can also deal 
with arbitrary covariance matrices, for which the 
runtime is a bit slower. 

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/citation
   user/api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/getting_started
   tutorials/soho

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
