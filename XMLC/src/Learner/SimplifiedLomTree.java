package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import preprocessing.FeatureHasherFactory;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;
import threshold.ThresholdTuning;
import util.CompleteTree;
import util.DecisionTree;
import util.HashSetBasedDecisionTree;
import util.MasterSeed;
import util.PointerBasedDecisionTree;

public class SimplifiedLomTree extends MLLRFHR {
	
	private static Logger logger = LoggerFactory.getLogger(SimplifiedLomTree.class);
	
	
	DecisionTree tree = null;
	
	int treeSize = 2;
	int treeSizeFactor = 0;
	
	HashMap<Integer, Double> bias = null;
	HashMap<Integer, Integer> Tarray = null; 
	HashMap<Integer, Double> scalararray = null;
	HashMap<Integer, Statistics> nodeStatistics = null;
	

	protected class Statistics {
		
		HashMap<Integer,Double> sumOfScoresForClass = new HashMap<>();
		HashMap<Integer,Integer> classDistribution = new HashMap<>();
		HashMap<Integer,Integer> numberOfExamplesUsedToTrainFromEachClass = new HashMap<>();
		HashMap<Integer,Double> expectedScoreForClass = new HashMap<>();
		double expectedScoreNominator = 0.0;
		double expectedScoreDenominator = 0.0;
		double expectedScore = 0.0;
		int denominator = 0;
		
	}
	
	public SimplifiedLomTree(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Learner: SimplifiedLomTree" );
		logger.info("#####################################################" );
		
		this.treeSizeFactor = Integer.parseInt(this.properties.getProperty("TreeSizeFactor", "1")); 
		logger.info("#### Tree size factor: " + this.treeSizeFactor);
		
		
	}
	
	public void allocateClassifiers(AVTable data) {
		
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		
		this.treeSize = this.treeSizeFactor * this.m - 1;
		
		this.tree = new HashSetBasedDecisionTree(this.treeSize);
		
		System.out.println( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		System.out.println("#####################################################" );
		System.out.println("#### Tree size: " + this.treeSize);	
		
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.treeSize);
		
		System.out.print( "Allocate the learners..." );

		this.w = new double[this.hd];
		
		bias = new HashMap<>(2*this.m-1);
		Tarray = new HashMap<>(2*this.m-1);
		scalararray = new HashMap<>(2*this.m-1);
		nodeStatistics = new HashMap<>(2*this.m-1);

		bias.put(0,0.0);
		Tarray.put(0, 1);
		scalararray.put(0, 1.0);
		nodeStatistics.put(0, new Statistics());		
				
		System.out.println( "Done." );
		
		
	}		
	
	
	@Override
	public void train(AVTable data) {
				
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for(int i = 0; i < traindata.n; i++) {
				
				int currIdx = indirectIdx.get(i);
				
				int label = traindata.y[currIdx][0];
				
				int node = 0;
				
				boolean end = false;
				
				while(!end) {
					
					Statistics ns = this.nodeStatistics.get(node);
					
					if(!ns.classDistribution.containsKey(label)) {
						ns.classDistribution.put(label, 0);
						ns.sumOfScoresForClass.put(label, 0.0);
						ns.numberOfExamplesUsedToTrainFromEachClass.put(label, 0);
						ns.expectedScoreForClass.put(label, 0.0);
					}
					ns.classDistribution.put(label, ns.classDistribution.get(label) + 1);
					ns.denominator++;
					
					if(this.tree.isLeaf(node)) {
						if( ns.classDistribution.size() >= 2 && this.tree.getCurrentNumberOfLeaves() < this.tree.getNumberOfLeaves()) {
							if(this.tree.expandTree(node)) {
								System.out.println(node);	
								for(int ch : this.tree.getChildNodes(node)) {
									nodeStatistics.put(ch, new Statistics());
									this.Tarray.put(ch, 1);
									this.scalararray.put(ch, 1.0);
									this.bias.put(ch, 0.0);
								}
							}
						}
						end = true;
					}
					if(!this.tree.isLeaf(node)) {
						double response;
						if(ns.expectedScore > ns.expectedScoreForClass.get(label)) {
							response = -1;
						}	else {
							response = 1;
						}
						updatedPosteriors(currIdx, node, -(response - getPartialPosteriors(traindata.x[currIdx], node)));
						double posterior = getPartialPosteriors(traindata.x[currIdx], node);
					
						ns.numberOfExamplesUsedToTrainFromEachClass.put(label, ns.numberOfExamplesUsedToTrainFromEachClass.get(label) + 1);
						ns.sumOfScoresForClass.put(label, ns.sumOfScoresForClass.get(label) + posterior);
						ns.expectedScoreForClass.put(label, ns.sumOfScoresForClass.get(label) / (double) ns.numberOfExamplesUsedToTrainFromEachClass.get(label));
						ns.expectedScoreNominator += posterior;
						ns.expectedScoreDenominator += 1;
						ns.expectedScore = ns.expectedScoreNominator / (double) ns.expectedScoreDenominator;
						
						
						if(posterior > 0) {
							node = this.tree.getChildNodes(node).get(0);
						}
						else {
							node = this.tree.getChildNodes(node).get(1);
						}
					}
				}
								
				if ((i % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					//System.out.println("Weight: " + this.w[0].get(0) );
				}
			}

			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
		}
		
		int zeroW = 0;
		double sumW = 0;
		int maxNonZero = 0;
		int index = 0;
		for(double weight : w) {
			if(weight == 0) zeroW++;
			else maxNonZero = index;
			sumW += weight;
			index++;
		}
		System.out.println("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last nonzero): " + w.length + ", " + zeroW + ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW)/(double) w.length + ", " + sumW + ", " + maxNonZero);

	}
	
	
	protected void updatedPosteriors( int currIdx, int label, double inc) {
		
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * Tarray.get(label));
		this.Tarray.put(label,  Tarray.get(label) + 1);
		this.scalararray.put(label, this.scalararray.get(label)*  (1 + this.learningRate * this.lambda));
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient =  this.scalararray.get(label) * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
			
		}
		
		double gradient =  this.scalararray.get(label) * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias.put(label, this.bias.get(label) - update); 
	}
	
	
	public double getPartialPosteriors(AVPair[] x, int label) {
		
		double posterior = 0.0;
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value *sign) * (1/this.scalararray.get(label)) * this.w[hi];
		}
		
		posterior += (1/this.scalararray.get(label)) * this.bias.get(label); 
		posterior = 2*s.value(posterior) - 1;		
		
		return posterior;
	}
	
	public int findLeaf(AVPair[] x) {
	
		int node = 0;
		
		while(!this.tree.isLeaf(node)) {
			
			AbstractLearner.numberOfInnerProducts++;
			
			double posterior = getPartialPosteriors(x, node);
			
			if(posterior > 0) {
				node = this.tree.getChildNodes(node).get(0);
			}
			else {
				node = this.tree.getChildNodes(node).get(1);
			}
		
		}
		
		return node;
	}
	
	
	@Override
	public double estimateProbability(AVPair[] x, int label) {
		int node = this.findLeaf(x);
		return this.nodeStatistics.get(node).classDistribution.get(label) / (double) this.nodeStatistics.get(node).denominator;
	}

	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {

		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();
		int  node = this.findLeaf(x);
		//System.out.println(this.nodeStatistics.get(node).classDistribution.toString());
		double denominator = (double) this.nodeStatistics.get(node).denominator;
		for(int label : this.nodeStatistics.get(node).classDistribution.keySet()) {
			positiveLabels.add(new EstimatePair(label, this.nodeStatistics.get(node).classDistribution.get(label) / denominator));
			if(positiveLabels.size() > k){			
				positiveLabels.pollLast();
			}
		}
		//System.out.println(positiveLabels.toString());
		return positiveLabels;
	}

}
