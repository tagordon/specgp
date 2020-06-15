from setuptools import setup 

setup(name='SpecGP', 
      version='0.1',
      description='Gaussian processes for multi-wavelength and spectral observations',
      url='http://github.com/tagordon/SpecGP',
      author='Tyler Gordon',
      author_email='tagordon@uw.edu', 
      license='MIT',
      packages=['SpecGP'],
      install_requires=['numpy',
                        'exoplanet',
                        'pymc3',
                        'theano'],
      zip_safe=False)
