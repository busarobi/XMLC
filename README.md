# XMLC [![Build Status](https://travis-ci.org/busarobi/XMLC.svg?branch=master)](https://travis-ci.org/busarobi/XMLC)
Extreme Classification: Probabilistic Label Tree

This package implements the F-measure optimization method for Extreme Multi-Label Classification by using __Probabilistic Label Tree (PLT)__ proposed by:

@inproceedings{JasinskaDBPKH16,


  author    = {Kalina Jasinska and
               Krzysztof Dembczynski and
               R{\'{o}}bert Busa{-}Fekete and
               Karlson Pfannschmidt and
               Timo Klerx and
               Eyke H{\"{u}}llermeier},
  title     = {Extreme F-measure Maximization using Sparse Probability Estimates},
  booktitle = {Proceedings of the 33nd International Conference on Machine Learning,
               {ICML} 2016, New York City, NY, USA, June 19-24, 2016},
  pages     = {1435--1444},
  year      = {2016},
  url       = {http://jmlr.org/proceedings/papers/v48/jasinska16.html},
 }  
 
 
 The packege consists of three use cases:
 
 1. Train PLT
 2. Output posteriors based the trained model
 3. Tune thresholds for optimizing the Macro F-measure
 4. Compute the prediction based on a model and corresponding thresholds that were validated for macro F-measreu
 
 