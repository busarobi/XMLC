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
import util.DecisionTreeNode;
import util.DecisionTreeNodePlus;
import util.HashSetBasedDecisionTree;
import util.MasterSeed;
import util.PointerBasedDecisionTree;
import util.PointerBasedDecisionTreePlus;

public class LomTreePlus extends MLLRFHR {
	
	private static final long serialVersionUID = 4576116358399421362L;

	PointerBasedDecisionTreePlus tree = null;
	
	private static Logger logger = LoggerFactory.getLogger(LomTreePlus.class);
	
	int treeSizeFactor = 1;
	int resistance = 4;
	int treeSize = 0;
	
	
	public LomTreePlus(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		System.out.println("#####################################################" );
		System.out.println("#### Learner: LomTreePlus" );
		System.out.println("#####################################################" );
		
		this.resistance = Integer.parseInt(this.properties.getProperty("Resistance", "4")); 
		logger.info("#### Resistance: " + this.resistance );
		
		this.treeSizeFactor = Integer.parseInt(this.properties.getProperty("TreeSizeFactor", "1")); 
		logger.info("#### Tree size factor: " + this.treeSizeFactor);
		
	}
	
	public void allocateClassifiers(AVTable data) {
		
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		
		this.treeSize = this.treeSizeFactor * this.m - 1;
		
		this.tree = new PointerBasedDecisionTreePlus((int) (this.treeSize));
		
		System.out.println( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		System.out.println("#####################################################" );
		System.out.println("#### Tree size: " + this.treeSize);	
		
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.treeSize);
		
		System.out.print( "Allocate the learners..." );

		this.w = new double[this.hd];
				
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
				
				DecisionTreeNodePlus node = this.tree.getRoot();
				
				boolean end = false;
				
				while(!end) {
					
					node.updateNode(label);
					
					if(node.isLeaf()) {
						if( node.getNumberOfClasses() >= 2) {
							
							if( this.tree.getSize() < this.tree.getMaximalSize() || (node.getC() - node.getMaxFrequency()) > (this.resistance * (this.tree.getC() + 1)) ) {
								
								if ( this.tree.getSize() < this.tree.getMaximalSize() ) {
									if(this.tree.expandTree(node)) {
										System.out.println(node.getNodeIndex());	
									}
								}
								else {
									this.tree.swap(node);
								}
								node.getLeftChild().setC((int) Math.floor(node.getC() / 2.0));
								node.getRightChild().setC(node.getC() - node.getLeftChild().getC());
								//tree.updateC(node.getLeftChild());
							}
						}
						//end = true;
					}
					if(!node.isLeaf()) {
						
						double posterior = getPartialPosteriors(traindata.x[currIdx], node);
						node.updateStatistics(label, posterior);
						
						//System.out.println("LomTreePlus: " + (node.getExpectedScore(label) - node.getExpectedScore()));
						//System.out.println("theta_y: " + node.getExpectedScore(label));
						//System.out.println("E(theta): " + node.getExpectedScore());
						
						double margin = (node.getExpectedScore(label) - node.getExpectedScore());
						
						updatedPosteriors(currIdx, node, margin);
												
						if(node.getExpectedScore(label) - node.getExpectedScore() >= 0) {
							node = node.getLeftChild();
						}
						else {
							node = node.getRightChild();
						}
					} else {
						node.setC(node.getC() + 1);
						this.tree.updateC(node);
						//System.out.println("Node: " + node.getNodeIndex() + " " + node.getC() + " " + this.tree.getC());
						break;
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
		
		System.out.println("Number of swaps: " + this.tree.getNumOfSwaps());
		System.out.println("Tree height: " + this.tree.height());
		//System.out.println("Tree:");
		//System.out.println(this.tree.toString());
		
	}
	
	
	protected void updatedPosteriors( int currIdx, DecisionTreeNodePlus node, double inc) {
		
		int T = node.getT(); 
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * node.getT());
		node.setT(T + 1);
		node.setScalar(node.getScalar() *  (1 + this.learningRate * this.lambda));
		
		int nodeIndex = node.getNodeIndex();
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(nodeIndex, traindata.x[currIdx][i].index);
			int sign = fh.getSign(nodeIndex, traindata.x[currIdx][i].index);
			
			double gradient =  node.getScalar() * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
			
		}
		
		double gradient =  node.getScalar() * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		node.setBias(node.getBias() - update); 
	}
	
	
	public double getPartialPosteriors(AVPair[] x, DecisionTreeNodePlus node) {
		
		double posterior = 0.0;
		
		int nodeIndex = node.getNodeIndex();
			
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(nodeIndex,  x[i].index); 
			int sign = fh.getSign(nodeIndex, x[i].index);
			posterior += (x[i].value *sign) * (1/node.getScalar()) * this.w[hi];
		}
		
		posterior += (1/node.getScalar()) * node.getBias(); 
		posterior = 2*s.value(posterior)-1;		
		
		return posterior;
	}
	
	public DecisionTreeNodePlus findLeaf(AVPair[] x) {
	
		DecisionTreeNodePlus node = this.tree.getRoot();
		
		while(!node.isLeaf()) {
			
			AbstractLearner.numberOfInnerProducts++;
			
			double posterior = getPartialPosteriors(x, node);
			
			System.out.println("Posterior" + posterior);
			
			if(posterior >= 0) {
				node = node.getLeftChild();
			}
			else {
				node = node.getRightChild();
			}
		
		}
		
		return node;
	}
	
	
	@Override
	public double estimateProbability(AVPair[] x, int label) {
		DecisionTreeNodePlus node = this.findLeaf(x);
		return node.getProbability(label); 
	}

	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {

		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();
		DecisionTreeNodePlus  node = this.findLeaf(x);
		//System.out.println(node.getClassDistribution().toString());
		for(int label : node.getLabels()) {
			positiveLabels.add(new EstimatePair(label, node.getProbability(label)));
			if(positiveLabels.size() > k){			
				positiveLabels.pollLast();
			}
		}
		System.out.println(positiveLabels.toString());
		return positiveLabels;
	}

}
