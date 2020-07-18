from setuptools import setup 

setup(name='specgp', 
      version='0.1',
      description='Gaussian processes for multi-wavelength and spectral observations',
      url='http://github.com/tagordon/specgp',
      author='Tyler Gordon',
      author_email='tagordon@uw.edu', 
      license='MIT',
      packages=['specgp'],
      install_requires=['numpy',
                        'scipy'
                        'exoplanet',
                        'pymc3',
                        'theano'],
      zip_safe=False)
