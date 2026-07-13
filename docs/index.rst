.. DiffRACC documentation master file, created by
   sphinx-quickstart on Mon Jul 13 17:10:37 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DiffRACC
========

Welcome to ``DiffRACC``'s documentation! DiffRACC stands for **Diff**usion-based **R**adio-loud **A**GN **C**ompleteness **C**orrections, and is a Python package designed to facilitate the usage of a diffusion model to generate synthetic radio galaxy images for the purpose of estimated completeness corrections for resolved sources in radio astronomy surveys. The package provides a suite of tools for preprocessing data, training diffusion models, and evaluating their performance, and along with many plotting utilities.

``DiffRACC`` is developed and maintained by Ashley Parr and Luna Greenberg, and formed the code underlining their shared MPhys project at the University of Manchester. Please refer to the `GitHub repository <https://github.com/solalunara/mphys-rlf>`_ for the latest updates, issues, and contributions, and feel free to make your own issues, forks, and pull requests as appropriate.


Accreditation
+++++++++++++
Please cite the `DOI <PLACEHOLDER>`_ and the paper describing the method if you make use of this software in your research.

Acknowledgements
++++++++++++++++
``DiffRACC`` is a package that creates a pipeline around a diffusion-model architecture. The code for this architecture comes directly from Vičánek Martínez et al. (2024), who in turn were implementing the EDM Preconditioning diffusion model from Karras et al. (2023). If you wish to cite the model architecture, you are invited to cite those papers, which can be found under the references below. The training procedure and output manager is also a slightly modified version of the files from Vičánek Martínez et al. (2024).

The authors would also like to thank Nutthawara Buatthaisong for her contributions of the concept and skeleton code for this method of estimating completeness correction and resultant radio luminosity function, and Anna Scaife, for her supervision and advice during the development of this project.


User Guides
+++++++++++
.. toctree::
   :maxdepth: 2


Public API
++++++++++

.. toctree::
   :maxdepth: 2



Indices and tables
++++++++++++++++++

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

