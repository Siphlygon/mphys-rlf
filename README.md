# DiffRACC
**Diff**usion-based **R**adio-loud **A**GN **C**ompleteness **C**orrections
By Ashley Parr and Luna Greenberg, University of Manchester

For installation instructions, tutorials, and detailed documentation, start [here](http://diffracc.readthedocs.io).

<!-- ![Build Status](https://github.com/Siphlygon/SEAWRD/actions/workflows/python-package.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/Siphlygon/SEAWRD/badge.svg?branch=main)](https://coveralls.io/github/Siphlygon/SEAWRD?branch=main)
[![Documentation Status](https://readthedocs.org/projects/SEAWRD/badge/?version=latest)](http://seawrd.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/seawrd)
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/) -->

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20869608.svg)](https://doi.org/10.5281/zenodo.20869608)
![GitHub License](https://img.shields.io/github/license/Siphlygon/SEAWRD) -->


## Motivation

Active Galactic Nuclei (AGN) are thought to be an essential component in the evolution of their host galaxy, and galaxy formation models must be able to accurately replicate the current distribution of AGN we see today in the universe. In order to account for this, population statistics of AGN are often inputs to semi-analytical models, with an important example being the radio luminoisty function (RLF).

These statistics are calculated from sources catalogued in wide-area surveys, which have a fundamental level of incompleteness. Accurately interpreting these statistics requires a quantative understanding of what sources are missed in these surveys, due to system noise, source finding algorithms, or other effects, lest we mistake the observe distribution for the true sky distribution. This is the role of a completeness correction, and is especially important for low-flux sources (the low-luminosity or low-redshift end of the RLF).

In order to determine which types of sources are missed by surveys, we must generate artificial sources of different properties and run appropriate detection pipelines. Contemporary approaches usually use (beam correlated) point sources or Gaussian sources of varying sizes; however, these are not good approximations for resolved radio galaxy sources, which have discernible extended structure. For these sources, completeness is also a function of size and morphology, and therefore a method of generating realistic radio galaxy images is desired. This becomes increasingly desirable with new radio surveys, which will be able to resolve far more radio galaxy sources than ever before.

This is what **Diff**usion-based **R**adio-loud **A**GN **C**ompleteness **C**orrections is for! It takes radio galaxy cutouts from the radio-optical cross-matched catalogue (Hardcastle et al. 2023) of the LOFAR Two-metre Sky Survey Data Release 2 (LoTSS-DR2; Shimwell et al. 2022), creates a pre-processed dataset which can be used to train a diffusion model with the provided architecture and training scripts. The same model can then be used to create generated datasets of specific fluxes using classier-free guidance, which can then be used to estimate a completeness correction using the provided code. This can be compared directly to similar results in the literature, and the impact can be further explored by seeing its effects on the estimated RLAGN RLF.

## Acknowledgements

DiffRACC is a package that creates a pipeline around a diffusion-model architecture. However, the code for this architecture comes directly from Vičánek Martínez et al. (2024), who in turn was implementing the EDM Preconditioning diffusion model from Karras et al. (2023). If you wish to cite the model architecture, you are invited to cite those papers, which can be found under references. The training procedure and output manager is also a slightly modified version of the files from Vičánek Martínez et al. (2024).

The authors would also like to thank Nutthawara Buatthaisong for her contributions of the concept and skeleton code for this method of estimating completeness correction and resultant radio luminosity function, and Anna Scaife, for her supervision and advice during the development of this project.

## Attribution

Please cite this paper if this package is used in your research. [PLACEHOLDER] 

## References

M. J. Hardcastle et al., “The LOFAR Two-Metre Sky Survey: VI. Optical identifications for the second data release” Astronomy and Astrophysics, vol. 678, p. A151, Sep. 2023, doi: [10.1051/0004-6361/202347333](https://doi.org/10.1051/0004-6361/202347333).

T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of Diffusion-Based Generative models,” arXiv, Jun. 2022, doi: [10.48550/arxiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364).

T. W. Shimwell et al., “The LOFAR Two-metre Sky Survey: V. Second data release” Astronomy and Astrophysics, vol. 659, p. A1, Jan. 2022, doi: [10.1051/0004-6361/202142484](https://doi.org/10.1051/0004-6361/202142484).

T. Vičánek Martínez, N. Baron Perez, and M. Brüggen, “Simulating images of radio galaxies with diffusion models,” Astronomy and Astrophysics, vol. 691, p. A360, Oct. 2024, doi: [10.1051/0004-6361/202451429](https://doi.org/10.1051/0004-6361/202451429).

