# XMLC [![Build Status](https://travis-ci.org/busarobi/XMLC.svg?branch=master)](https://travis-ci.org/busarobi/XMLC)
Extreme Classification: Probabilistic Label Tree

This package implements the F-measure optimization method for Extreme Multi-Label Classification by using __Probabilistic Label Tree (PLT)__ proposed by:

Kalina Jasinska, Krzysztof Dembczynski, Ro}bert Busa-Fekete, Karlson Pfannschmidt, Timo Klerx and Eyke Hullermeier, __Extreme F-measure Maximization using Sparse Probability Estimates__, *Proceedings of the 33nd International Conference on Machine Learning, {ICML} 2016, New York City, NY, USA, June 19-24,* p.1435-1444, 2016, [pdf](http://jmlr.org/proceedings/papers/v48/jasinska16.html)
 
 
 The package consists of three use cases:
 
 1. Train PLT
 2. Output posteriors based the trained model
 3. Tune thresholds for optimizing the Macro F-measure
 4. Compute the prediction based on a model and corresponding thresholds that were validated for macro F-measreu
 
 