package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import Learner.step.StepFunction;
import preprocessing.FeatureHasherFactory;
import preprocessing.UniversalHasher;
import util.CompleteTree;

public class PLTFHKary extends MLLRFH {
	private static final long serialVersionUID = 7616508564698690408L;

	private static Logger logger = LoggerFactory.getLogger(PLTFHKary.class);

	transient protected int t = 0;
	
	transient CompleteTree tree = null;
	
	protected int k = 2;
	
	protected Object readResolve(){
		this.tree = new CompleteTree(this.k, this.m);
		this.t = this.tree.getSize(); 
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		return this;
	}
	
	public PLTFHKary(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		
		logger.info("#####################################################" );
		logger.info("#### Leraner: PLTFTHKary" );
		
		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		logger.info("#### k: " + this.k );

	}

	@Override
	public void allocateClassifiers(AVTable data) {
		
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		
		
		this.tree = new CompleteTree(this.k, this.m);
		
		this.t = this.tree.getSize(); 

		logger.info( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info( "#### Num. of inner node of the trees: " + this.t  );
		logger.info("#####################################################" );
			
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}
		
		logger.info( "Done." );
	}
		
	@Override
	public void train(AVTable data) {
		
		this.T = 1;
				
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
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

				//logger.info("Negative tree indices: " + negativeTreeIndices.toString());

				for(int j:positiveTreeIndices) {

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(1.0 - posterior); 

					updatedPosteriors(currIdx, j, inc);
				}

				for(int j:negativeTreeIndices) {

					if(j >= this.t) logger.info("ALARM");

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(currIdx, j, inc);
				}

				this.T++;

				if ((i % 100000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					//logger.info("Weight: " + this.w[0].get(0) );
					logger.info("Scalar: " + this.scalar);
				}
			}

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
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
		logger.info("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last nonzero): " + w.length + ", " + zeroW + ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW)/(double) w.length + ", " + sumW + ", " + maxNonZero);
	}


	public double getPartialPosteriors(AVPair[] x, int label) {
		double posterior = super.getPosteriors(x, label);
		//logger.info("Partial posterior: " + posterior + " Tree index: " + label);
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
		//if(posterior > 0.5) logger.info("Posterior: " + posterior + "Label: " + label);
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

		//logger.info("Predicted labels: " + positiveLabels.toString());

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

		//logger.info("Predicted labels: " + positiveLabels.toString());

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
	
		//logger.info("Predicted labels: " + positiveLabels.toString());

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
		//	logger.info( "Threshold: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		
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

		//logger.info("Predicted labels: " + positiveLabels.toString());

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
	
		
	
	public void load(String fname) {
		super.load(fname);
		this.t = 2 * this.m - 1;
		this.fh = new UniversalHasher(this.fhseed, this.hd, this.t);
	}
	
	
	
}
