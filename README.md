# astrostatistics

Repository for the Individual Research Project on "Machine Learning for Systematics Contamination Modelling in Galaxy Surveys".

### Abstract

By measuring the intrinsic clustering of galaxies across our sky, cosmologists attempt to constrain important cosmological parameters. Problematically, the distribution of galaxies that we find in large-scale galaxy catalogues is often correlated to background contaminants (_imaging systematics_) that have no underlying cosmological cause. In the presence of this contamination, unbiased measurements of intrinsic galaxy clustering are impossible to conduct. As such, modelling and removing the impact of these imaging systematics on the perceived distribution of galaxies is imperative to scientific progress in cosmology.

Extracting and removing the impact of these systematics is no simple task and is subject to active research in cosmology. Critically, the level of contamination in galaxy catalogues is a complex function of several overlapping exposures taken by telescopes, that is, the level of contamination is determined by unordered sets of variable size. The state of the art manually aggregates these unordered sets into single-dimensional feature vectors, before applying linear regression to model the systematics contamination function f(sys)\\

This paper presents two improvements to the state of the art. First, the function $$f(\mathrm{sys})$$ is approximated using a feed-forward neural network instead of linear regression. Secondly, a DeepSets architecture is implemented, thus bypassing manual aggregation of systematics by directly learning $f(\mathrm{sys})$ on the variable-sized unordered sets of overlapping exposures. The state of the art, neural network and DeepSets architectures are trained, tested and evaluated on a catalogue of 115 million galaxies specifically generated for this research project. Both the neural network and the DeepSets architecture substantially outperform the state of the art linear regression methodology. The neural network shows the highest consistency among all methods of contamination mitigation, while the DeepSets architecture outperforms the other methods in the most contaminated areas of the catalogue. Especially the DeepSets' capacity of learning functions on variable-sized unordered sets promises to find wider application on a plethora of other data sources in computational cosmology.



Implemented in partial fulfillment of the requirements for the MSc degree inComputing Science of Imperial College London



[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
