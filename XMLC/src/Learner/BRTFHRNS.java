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
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Queue;
import java.util.Random;
import java.util.TreeSet;

import org.apache.commons.math3.analysis.function.Sigmoid;

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
import util.MasterSeed;

public class BRTFHRNS extends MLLRFHRNS {
	
	protected int t = 0;
		 
	CompleteTree tree = null;
	
	protected int k = 2;
	
	double threshold = 0.05;
	
	
	public BRTFHRNS(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		System.out.println("#####################################################" );
		System.out.println("#### Learner: BRTFHRNS" );
		
		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		System.out.println("#### k: " + this.k );
		
		System.out.println("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {

		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		
		this.tree = new CompleteTree(this.k, this.m);
		
		this.t = this.tree.getSize();

		System.out.println( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		System.out.println( "#### Num. of inner node of the trees: " + this.t  );
		System.out.println("#####################################################" );
			
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		
		System.out.print( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}

		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
		
		System.out.println( "Done." );
		
		numOfUpdates = new int[this.t]; 
		numOfPositiveUpdates = new int[this.t];
		numOfNegativeUpdates = new int[this.t];
		
		this.contextChange = new double[this.t];
 
		 
	}
	
		
	@Override
	public void train(AVTable data) {
						
		HashMap<Integer,Integer> positiveLabels = new HashMap<>(); 
		HashSet<Integer> negativeLabels = new HashSet<>();
		
		
		Random random = new Random(1);
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {
				
				int currIdx = indirectIdx.get(i);
				
				int numOfNegToSample = traindata.y[currIdx].length * this.samplingRatio;
				
				positiveLabels.clear();
				negativeLabels.clear();
				
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					
					int treeIndex = this.tree.getTreeIndex(traindata.y[currIdx][j]);
					
					/*positiveLabels.put(treeIndex, 1); 
					numOfUpdates[treeIndex]++;
					numOfPositiveUpdates[treeIndex]++;
					*/
					
					while(treeIndex >= 0) {
						
						Integer value = positiveLabels.get(treeIndex);
						
						if(value == null) {
							positiveLabels.put(treeIndex, 1);
						} else {
							positiveLabels.put(treeIndex, value + 1);
						}
						
						numOfUpdates[treeIndex]++;
						numOfPositiveUpdates[treeIndex]++;

						treeIndex = this.tree.getParent(treeIndex);

					}
					
				}
				
				int numOfNegativeLabels = 0;
				
				while(numOfNegativeLabels < numOfNegToSample && numOfNegativeLabels < this.m - numOfNegToSample) {
					
					int treeIndex = this.tree.getTreeIndex(random.nextInt(this.m));
					
					if(negativeLabels.contains(treeIndex) || positiveLabels.containsKey(treeIndex))
						continue; 
					else
						numOfNegativeLabels++;
					
					/*if(!positiveLabels.containsKey(treeIndex)) {
						if(!negativeLabels.contains(treeIndex)) {
							negativeLabels.add(treeIndex);
							numOfUpdates[treeIndex]++;
							numOfNegativeUpdates[treeIndex]++;
							numOfNegativeLabels++;
						}
					}*/
					
					do {
					
						negativeLabels.add(treeIndex);
						numOfUpdates[treeIndex]++;
						numOfNegativeUpdates[treeIndex]++;
						treeIndex = this.tree.getParent(treeIndex);
					
					}	while(treeIndex >= 0 && !positiveLabels.containsKey(treeIndex));
				}
				
				//System.out.println("Positive labels: " + positiveLabels.toString());
				
				//System.out.println("Negative labels: " + negativeLabels.toString());
				
				for(int j:positiveLabels.keySet()) {

					double posterior = getUncalibratedPosteriors(traindata.x[currIdx],j);
					double inc = -(1.0 - posterior); 
				
					updatedPosteriors(currIdx, j, inc, positiveLabels.get(j));
				}

				for(int j:negativeLabels) {

					double posterior = getUncalibratedPosteriors(traindata.x[currIdx],j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(currIdx, j, inc);
				}			
				
				//if(i == 5) System.exit(1);
				
				if ((i % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					//System.out.println("Weight: " + this.w[0].get(0) );
					System.out.println("Scalar: " + this.scalararray[0]);
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
		
		for(int i = 0; i < this.t; i++) {
			this.contextChange[i] = this.computeContextChange(this.numOfPositiveUpdates[i], this.numOfUpdates[i], (this.traindata.n * this.epochs));
//			System.out.println(i + ": " + numOfUpdates[i] + " " + (this.epochs * this.traindata.n) + " " + (((double) this.numOfUpdates[i]/(double) (this.epochs * this.traindata.n))));
		}

	}


	protected void updatedPosteriors( int currIdx, int label, double inc, double weight) {
	
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);		
			this.w[index] -= weight * update; 
			
		}
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= weight * update;

	}

	
	@Override
	public double getPosteriors(AVPair[] x, int label) {

		int treeIndex = this.tree.getTreeIndex(label);
		double posterior = super.getPosteriors(x, treeIndex);
		//System.out.println(label + "\t" + treeIndex + "\t" + posterior);
		return posterior;
	
	}
	
	
	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> positiveLabels = new HashSet<Integer>();
	    
		Queue<Integer> queue = new LinkedList<Integer>();

		queue.add(0);

		while(!queue.isEmpty()) {

			int node = queue.poll();

			double currentP = this.getUncalibratedPosteriors(x, node);//super.getPosteriors(x, node);

			if(currentP >= this.thresholds[node]) {

				if(!this.tree.isLeaf(node)) {
					
					for(int childNode: this.tree.getChildNodes(node)) {
						queue.add(childNode);
					}
					
				} else {

					positiveLabels.add(this.tree.getLabelIndex(node));

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	public TreeSet<EstimatePair> getCandidateLabels(AVPair[] x, double threshold, int k) {

		TreeSet<EstimatePair> candidateLabels = new TreeSet<EstimatePair>();
		Queue<Integer> queue = new LinkedList<Integer>();
		queue.add(0);

		while(!queue.isEmpty()) {
			int node = queue.poll();
			double p = super.getPosteriors(x, node); 
			if(!this.tree.isLeaf(node)) {
				if(p >= threshold) {
					for(int childNode: this.tree.getChildNodes(node)) {
						queue.add(childNode);
					}
				} 
			} else {
				candidateLabels.add(new EstimatePair(this.tree.getLabelIndex(node), p));
				if(candidateLabels.size() > k){			
					candidateLabels.pollLast();
				}
			}
		}
		//System.out.println("Candidate labels: " + candidateLabels.toString());
			
		return candidateLabels;
	}

	
	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		
		
		TreeSet<EstimatePair> candidateLabels = getCandidateLabels(x, this.threshold, k);
		
		return candidateLabels;
		
	}

	
	
	@Override
	public void setThreshold(int label, double t) {
		
		int treeIndex = this.tree.getTreeIndex(label);
		this.thresholds[treeIndex] = t;
		
		while(treeIndex > 0) {
			treeIndex =  this.tree.getParent(treeIndex);
			double minThreshold = Double.MAX_VALUE;
			for(int childNode: this.tree.getChildNodes(treeIndex)) {
				minThreshold = this.thresholds[childNode] < minThreshold?  this.thresholds[childNode] : minThreshold;
			}	
			this.thresholds[treeIndex] = minThreshold;
		}
		
	}

	public void setThresholds(double t) {		
		for(int j = 0; j < this.thresholds.length; j++) {
			this.thresholds[j] = t;
		}		
	}
	
	public void setThresholds(double[] t) {
		
		for(int j = 0; j < t.length; j++) {
			this.thresholds[this.tree.getTreeIndex(j)] = t[j];
		}
		
		for(int j = this.tree.getNumberOfInternalNodes()-1; j >= 0; j--) {
			
			double minThreshold = Double.MAX_VALUE;
			for(int childNode: this.tree.getChildNodes(j)) {
				minThreshold = this.thresholds[childNode] < minThreshold?  this.thresholds[childNode] : minThreshold;
			}
			
			this.thresholds[j] = minThreshold; 
		}
	}

	
	@Override
	public void save(String fname) {
		
		//to be completed
		
	}

	@Override
	public void load(String fname) {
		
		//to be completed
	}
	
	
	
}
