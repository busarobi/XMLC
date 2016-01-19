package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;
import java.util.TreeSet;

import Data.AVPair;
import Data.NodePLT;
import Data.NodeComparatorPLT;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import preprocessing.FeatureHasherFactory;
import preprocessing.MaskHasher;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;
import threshold.ThresholdTuning;
import util.CompleteTree;
import util.MasterSeed;

public class PLTFHKary extends MLLRFH {
	
	protected int t = 0;
	
	CompleteTree tree = null;
	
	protected int k = 2;
	
	public PLTFHKary(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		
		System.out.println("#####################################################" );
		System.out.println("#### Leraner: PLTFTHKary" );
		
		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		System.out.println("#### k: " + this.k );

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
		
		System.out.println( "Done." );
	}
		
	@Override
	public void train(AVTable data) {
		
		this.T = 1;
				
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {

				this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.T);
				this.scalar *= (1 + this.learningRate * this.lambda);
				
				int currIdx = indirectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				for (int j = 0; j < traindata.y[currIdx].length; j++) {

					int treeIndex = this.tree.getTreeIndex(traindata.y[currIdx][j]); // + traindata.m - 1;
					positiveTreeIndices.add(treeIndex);

					while(treeIndex > 0) {

						treeIndex = (int) this.tree.getParent(treeIndex); // Math.floor((treeIndex - 1)/2);
						positiveTreeIndices.add(treeIndex);

					}
				}

				if(positiveTreeIndices.size() == 0) {

					negativeTreeIndices.add(0);

				} else {

					
					for(int positiveNode : positiveTreeIndices) {
						
						if(!this.tree.isLeaf(positiveNode)) {
							
							for(int childNode: this.tree.getChildNodes(positiveNode)) {
								
								if(!positiveTreeIndices.contains(childNode)) {
									negativeTreeIndices.add(childNode);
								}
								
							}
							
						}
						
					}
				}

				//System.out.println("Negative tree indices: " + negativeTreeIndices.toString());

				for(int j:positiveTreeIndices) {

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(1.0 - posterior); 

					updatedPosteriors(currIdx, j, inc);
				}

				for(int j:negativeTreeIndices) {

					if(j >= this.t) System.out.println("ALARM");

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(currIdx, j, inc);
				}

				this.T++;

				if ((i % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					//System.out.println("Weight: " + this.w[0].get(0) );
					System.out.println("Scalar: " + this.scalar);
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


	public double getPartialPosteriors(AVPair[] x, int label) {
		double posterior = super.getPosteriors(x, label);
		//System.out.println("Partial posterior: " + posterior + " Tree index: " + label);
		return posterior;
	}



	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 1.0;

		int treeIndex = this.tree.getTreeIndex(label);

		posterior *= getPartialPosteriors(x, treeIndex);

		while(treeIndex > 0) {

			treeIndex = (int) this.tree.getParent(treeIndex); //Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(x, treeIndex);

		}
		//if(posterior > 0.5) System.out.println("Posterior: " + posterior + "Label: " + label);
		return posterior;
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> positiveLabels = new HashSet<Integer>();

	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11,nodeComparator);

		queue.add(new NodePLT(0,1.0));

		while(!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if(currentP >= this.thresholds[node.treeIndex]) {

				if(!this.tree.isLeaf(node.treeIndex)) {
					
					for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}
					
				} else {

					positiveLabels.add(this.tree.getLabelIndex(node.treeIndex));

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	
	@Override
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(0,1.0));

		while(!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if(currentP > this.thresholds[node.treeIndex]) {

				if(!this.tree.isLeaf(node.treeIndex)) {
					
					for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}
					
				} else {

					positiveLabels.add(new ComparablePair( currentP, this.tree.getLabelIndex(node.treeIndex)));

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}
	
	
	@Override
	public int[] getTopkLabels(AVPair[] x, int k) {
		int[] positiveLabels = new int[k];
		int indi =0;

	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<>(11, nodeComparator);

		queue.add(new NodePLT(0,1.0));

		while(!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			
			if(!this.tree.isLeaf(node.treeIndex)) {
				
				for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP));
				}
				
			} else {
				positiveLabels[indi++] = this.tree.getLabelIndex(node.treeIndex);
			}
			
			if (indi>=k) { 
				break;
			}
		}
	
		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
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
		
		//for( int i=0; i < this.thresholds.length; i++ )
		//	System.out.println( "Threshold: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		
	}


	
	@Override
	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold) {

		HashSet<EstimatePair> positiveLabels = new HashSet<EstimatePair>();

	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(0,1.0));

		while(!queue.isEmpty()) {

			NodePLT node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if(currentP >= threshold) {

				if(!this.tree.isLeaf(node.treeIndex)) {
					
					for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
						queue.add(new NodePLT(childNode, currentP));
					}
					
				} else {
				
					positiveLabels.add(new EstimatePair(this.tree.getLabelIndex(node.treeIndex), currentP));

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}



	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {

		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();

	    int foundTop = 0;
	    
	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(0,1.0));
		
		while(!queue.isEmpty() && (foundTop < k)) {

			NodePLT node = queue.poll();

			double currentP = node.p;
			
			if(!this.tree.isLeaf(node.treeIndex)) {
				
				for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP * getPartialPosteriors(x, childNode)));
				}
				
			} else {
				
				positiveLabels.add(new EstimatePair(this.tree.getLabelIndex(node.treeIndex), currentP));
				foundTop++;
				
			}
		}

		return positiveLabels;
	}
	
		
	
	@Override
	public void loadmodel(String fname) {
		super.loadmodel(fname);
		this.t = 2 * this.m - 1;
		this.fh = new UniversalHasher(this.fhseed, this.hd, this.t);
	}
	
	
	
}
