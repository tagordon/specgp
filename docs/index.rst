.. SpecGP documentation master file, created by
   sphinx-quickstart on Mon Jul 13 21:52:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

specgp
==================================
*specgp* enables 2D Gaussian process 
computations in *exoplanet*. It provides 
a GP kernel term, ``KronTerm(term, alpha or R)`` which 
combines a 1D *celerite* kernel term with a 
matrix specifying the covariance for the second 
dimension. 

While there are many uses for multidimensional 
Gaussian processes, one of spetial relevance in 
astronomy is simultaneously modeling variability 
in multiband 
light curves. Models of this type can 
be specified in *specgp* by a 1D *celerite* term 
to specify the common temporal covariance for 
each band and a vector, :math:`\alpha`, specifying  
a scaling relationship between the variability 
amplitudes in each band. Models of this type 
can be computed in :math:`\mathcal{O}(NM)` time 
for :math:`N` observations taken in :math:`M` bands. 

*specgp* is also capable of computing 2D GP 
models with arbitrary covariance in the second 
dimension which can be specified with a user-supplied 
covariance matrix :math:`R`. In this case the runtime 
scales as the cube of the size of the second dimension. 

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
   tutorials/sums_of_terms
   tutorials/soho

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
