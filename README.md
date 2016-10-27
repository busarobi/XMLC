# XMLC [![Build Status](https://travis-ci.org/busarobi/XMLC.svg?branch=master)](https://travis-ci.org/busarobi/XMLC)
# Extreme Classification: Probabilistic Label Tree for F-measure optimization

This package implements the F-measure optimization method for Extreme Multi-Label Classification by using __Probabilistic Label Tree (PLT)__ proposed by:

Kalina Jasinska, Krzysztof Dembczynski, Robert Busa-Fekete, Karlson Pfannschmidt, Timo Klerx and Eyke Hullermeier, __Extreme F-measure Maximization using Sparse Probability Estimates__, *Proceedings of the 33nd International Conference on Machine Learning, (ICML'16), New York City, NY, USA, June 19-24,* p.1435-1444, 2016, [pdf](http://jmlr.org/proceedings/papers/v48/jasinska16.html)

The package consists of two moduls. The first modul trains the model and outputs the model which implements a PLT. The PLT model is able to provide sparse probability estimates, that is, it is able to compute the posteriors that exceed a predifened threshold in an efficient way. Building on this mechanism, the second modul implements some efficient threshold tuning algorithms for the posterios that optimizes the macro F-measure. For more detail, please read our paper cited above, or contact us! 

Next, we briefly describe how to get started.
 
 
Download and compile jar
========================


> git clone -b cleanup https://github.com/busarobi/XMLC

> mvn compile package 

Having executed these two commands, you should find a jar called **XMLC_PLT-jar-with-dependencies.jar** in the root directory of git project.
 
 
The package implements the following use cases:
=========================================
 
 1. **"-train"** Train PLT
 1. **"-eval"** Evaluate the model on a given test file
 2. **"-posteriors"** Output posteriors based on a model
 3. **"-tune"** Tune thresholds for optimizing the Macro F-measure
 4. **"-test"** Compute the prediction based on a model and corresponding thresholds that were validated for macro F-measure



1. Train PLT
============= 

The parameters are taken from a config file. There is a sample config file for training PLT on the RCV1 dataset that is available from the [Extreme Classification Repository](http://research.microsoft.com/en-us/um/people/manik/downloads/XC/XMLRepository.html).

As a fist step, please download this dataset from the repository. Next, set the path to the training file in ./examples/rcv1_train.config. And run the training method by 

> java -Xmx12G -jar XMLC_PLT-jar-with-dependencies.jar -train ./examples/rcv1_train.config 

The model file is saved to directory which is defined by the parameter called *ModelFile*.

2. Evaluate a model
===================

To evaluate a model, call the same jar by using **"-eval'** as second command line parameter. The parameter *TestFile* needs to be set to the path to the test file and the *InputModelFile* to the model file. For example, if one wants to evaluate the model build in the previous step, the following comment should be executed: 

> java -Xmx12G -jar XMLC_PLT-jar-with-dependencies.jar -eval ./examples/rcv1_eval.config

The model will be evaluated in terms of Precision@K where K = {1,2,3,4,5}. Other evaluation metric can be easily implemented in the class *IO.Evaluator*


3. Compute posteriors
======================

To compute the posteriors for a given dataset, one needs give **"-posterios'** as second command line parameter and then the config file as follows:

> java -Xmx12G -jar XMLC_PLT-jar-with-dependencies.jar -posteriors ./examples/rcv1_eval.config


  



