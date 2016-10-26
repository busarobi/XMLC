package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;
import java.util.TreeSet;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import preprocessing.FeatureHasher;
import preprocessing.FeatureHasherFactory;
import util.CompleteTree;
import util.MasterSeed;

public class PLT extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(PLT.class);
	transient protected int t = 0;	
	transient CompleteTree tree = null;
	
	protected int k = 2;

	transient protected int T = 1;
	transient protected AVTable traindata = null;
	
	
	transient protected FeatureHasher fh = null;
	protected String hasher = "Universal";
	protected int fhseed = 1;
	protected int hd;
	
	protected double[] bias;
	protected double[] w = null;
	
	transient protected int[] Tarray = null;	
	protected double[] scalararray = null;

	transient protected double gamma = 0; // learning rate
	transient protected int step = 0;
	transient Sigmoid s = new Sigmoid();
	transient protected double learningRate = 1.0;
	protected double scalar = 1.0;
	transient protected double lambda = 0.00001;
	transient protected int epochs = 1;	
	
	Random shuffleRand = null;	
	
	public PLT(Properties properties) {
		super(properties);
		shuffleRand = MasterSeed.nextRandom();
		
		System.out.println("#####################################################" );
		System.out.println("#### Learner: PLT" );
		// learning rate
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "1.0"));
		logger.info("#### gamma: " + this.gamma );

		// scalar
		this.lambda = Double.parseDouble(this.properties.getProperty("lambda", "1.0"));
		logger.info("#### lambda: " + this.lambda );

		// epochs
		this.epochs = Integer.parseInt(this.properties.getProperty("epochs", "30"));
		logger.info("#### epochs: " + this.epochs );

		// epochs
		this.hasher = this.properties.getProperty("hasher", "Universal");
		logger.info("#### Hasher: " + this.hasher );
		
		
		this.hd = Integer.parseInt(this.properties.getProperty("MLFeatureHashing", "50000000")); 
		logger.info("#### Number of ML hashed features: " + this.hd );
		

		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		logger.info("#### k (order of the tree): " + this.k );
		
		System.out.println("#####################################################" );

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
		
		
		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
	}

	protected ArrayList<Integer> shuffleIndex() {
		ArrayList<Integer> indirectIdx = new ArrayList<Integer>(this.traindata.n);
		for (int i = 0; i < this.traindata.n; i++) {
			indirectIdx.add(new Integer(i));
		}
		Collections.shuffle(indirectIdx, shuffleRand);
		return indirectIdx;
	}
	
	
	@Override
	public void train(AVTable data) {
		
		
				
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1), this.epochs );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {

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


	protected void updatedPosteriors( int currIdx, int label, double inc) {
			
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
			
		}
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
	}
	
	
	public double getPartialPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value *sign) * (1/this.scalar) * this.w[hi];
		}
		
		posterior += (1/this.scalar) * this.bias[label]; 
		posterior = s.value(posterior);		
		
		return posterior;

	}

	protected Object readResolve(){
		this.tree = new CompleteTree(this.k, this.m);
		this.t = this.tree.getSize(); 
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		return this;
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

}
