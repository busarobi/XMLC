# XMLC [![Build Status](https://travis-ci.org/busarobi/XMLC.svg?branch=master)](https://travis-ci.org/busarobi/XMLC)
Extreme Classification: Probabilistic Label Tree

This package implements the F-measure optimization method for Extreme Multi-Label Classification by using __Probabilistic Label Tree (PLT)__ proposed by:

Kalina Jasinska, Krzysztof Dembczynski, Robert Busa-Fekete, Karlson Pfannschmidt, Timo Klerx and Eyke Hullermeier, __Extreme F-measure Maximization using Sparse Probability Estimates__, *Proceedings of the 33nd International Conference on Machine Learning, (ICML'16), New York City, NY, USA, June 19-24,* p.1435-1444, 2016, [pdf](http://jmlr.org/proceedings/papers/v48/jasinska16.html)

 
Download and compile jar
========================


> git clone -b cleanup https://github.com/busarobi/XMLC
> mvn compile package 
> java -jar ./target/XMLC-0.0.1-SNAPSHOT-jar-with-dependencies.jar mode configfile 
 
The package consists of three use cases:
=========================================
 
 1. Train PLT
 2. Output posteriors based the trained model
 3. Tune thresholds for optimizing the Macro F-measure
 4. Compute the prediction based on a model and corresponding thresholds that were validated for macro F-measreu



1. Train PLT
============= 

The parameters are taken from config file. There is a sample config file for training PLT on the RCV1 dataset that is available from the [Extreme Classification Repository](http://research.microsoft.com/en-us/um/people/manik/downloads/XC/XMLRepository.html).

As a fist step, please download this dataset from the reopository. Next, set the path to the trainng file in ./examples/rcv1_train.config. And run the training method by 

>java -Xmx12G -jar XMLC_PLT-jar-with-dependencies.jar -train ./examples/rcv1_train.config 

The model file is saved in *./examples/model_ontrain.model*

