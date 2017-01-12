package Learner;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDouble;
import com.google.common.util.concurrent.AtomicDoubleArray;

import Data.AVPair;
import Data.EstimatePair;
import Data.Instance;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import IO.DataManager;
import preprocessing.FeatureHasherFactory;
import util.CompleteTree;
import util.HuffmanTree;
import util.PrecomputedTree;
import util.Tree;

public class ParallelDeepPLT extends PLT {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(ParallelDeepPLT.class);

	protected AtomicDoubleArray[] w = null;
	protected AtomicDoubleArray[] hiddenWeights = null;
	protected AtomicDouble[] bias = null;

	protected int numOfThreads = 4;

	protected int hiddendim = 100;
	protected String hiddenVectorsFile = null;

	transient protected AtomicInteger[] Tarray = null;
	protected AtomicDouble[] scalararray = null;

	transient protected AtomicInteger[] Tarrayhidden = null;
	protected AtomicDouble[] scalararrayhidden = null;

	public ParallelDeepPLT(Properties properties) {
		super(properties);

		System.out.println("#####################################################");
		System.out.println("#### Learner: ParallelDeepPLT");

		this.hiddendim = Integer.parseInt(this.properties.getProperty("hiddendim", "100"));
		logger.info("#### Number of hidden dimension: " + this.hiddendim);

		this.hiddenVectorsFile = this.properties.getProperty("hiddenvectorsFile", null);
		logger.info("#### hidden vectors file name " + this.hiddenVectorsFile);

		//
		this.numOfThreads = Integer.parseInt(this.properties.getProperty("numThreads", "4"));
		logger.info("#### num of threads: " + this.numOfThreads);

		System.out.println("#####################################################");

	}

	@Override
	public void allocateClassifiers(DataManager data) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();

		switch (this.treeType) {
		case CompleteTree.name:
			this.tree = new CompleteTree(this.k, this.m);
			break;
		case PrecomputedTree.name:
			this.tree = new PrecomputedTree(this.treeFile);
			break;
		case HuffmanTree.name:
			this.tree = new HuffmanTree(data, this.treeFile);
			break;
		default:
			System.err.println("Unknown tree type!");
			System.exit(-1);
		}
		this.t = this.tree.getSize();

		logger.info("#### Num. of labels: " + this.m + " Dim: " + this.d);
		logger.info("#### Num. of inner node of the trees: " + this.t);
		logger.info("#####################################################");
		
		
		this.allocateMemory(data);
		
	}

	
	public void allocateClassifiers(DataManager data, Tree tree) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();
		this.tree = tree;
		this.t = this.tree.getSize();

		logger.info("#### Num. of labels: " + this.m + " Dim: " + this.d);
		logger.info("#### Num. of inner node of the trees: " + this.t);
		logger.info("#####################################################");
		
		
		this.allocateMemory(data);
		
	}
	
	public double[][] getDeepRepresentation(){
		double[][] retval = new double[this.d][];
		for (int i = 0; i < this.d; i++) {
			int hi = fh.getIndex(1, i);
			retval[i] = new double[this.hiddendim];
			for (int j = 0; j < this.hiddendim; j++) {
				retval[i][j] = this.hiddenWeights[hi].get(j);
			}
		}
		return retval;
	}
	
	
	public void allocateMemory(DataManager data) {
		if (this.hasher.compareTo("Identical")==0) { 
			this.hd = this.d;
		}
		
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, 1);

		logger.info("Allocate the learners...");

		Random r = new Random();
		this.hiddenWeights = new AtomicDoubleArray[this.hd];
		for (int i = 0; i < this.hd; i++) {
			this.hiddenWeights[i] = new AtomicDoubleArray(this.hiddendim);
			for (int j = 0; j < this.hiddendim; j++) {
				this.hiddenWeights[i].set(j, 2.0 * r.nextDouble() - 1.0);
			}
		}

		this.w = new AtomicDoubleArray[this.t];
		for (int i = 0; i < this.t; i++) {
			this.w[i] = new AtomicDoubleArray(this.hiddendim);
			for (int j = 0; j < this.hiddendim; j++) {
				this.w[i].set(j, 2.0 * r.nextDouble() - 1);
			}
		}

		this.thresholds = new double[this.t];
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}

		this.bias = new AtomicDouble[this.t];
		for (int i = 0; i < this.t; i++) {
			this.bias[i] = new AtomicDouble(0.0);
		}

		this.Tarray = new AtomicInteger[this.t];
		this.scalararray = new AtomicDouble[this.t];
		for (int i = 0; i < this.t; i++) {
			this.Tarray[i] = new AtomicInteger(1);
			this.scalararray[i] = new AtomicDouble(1.0);
		}

		this.Tarrayhidden = new AtomicInteger[this.hd];
		this.scalararrayhidden = new AtomicDouble[this.hd];
		for (int i = 0; i < this.hd; i++) {
			this.Tarrayhidden[i] = new AtomicInteger(1);
			this.scalararrayhidden[i] = new AtomicDouble(1.0);
		}

		logger.info("Done.");
	}
	
	
	
	@Override
	public void train(DataManager data) {
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1), this.epochs);

			// UpdateThread processingThread = new UpdateThread(data);
			// processingThread.run();

			UpdateThread[] processingThreads = new UpdateThread[this.numOfThreads];

			ExecutorService executor = Executors.newFixedThreadPool(this.numOfThreads);
			for (int i = 0; i < this.numOfThreads; i++) {
				processingThreads[i] = new UpdateThread(data, (ep * this.numOfThreads) + i);
				executor.execute(processingThreads[i]);
			}

			executor.shutdown();

			while (!executor.isTerminated()) {
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}

			for (int i = 0; i < this.numOfThreads; i++) {
				logger.info("Thread: " + processingThreads[i].getID() + " --- Num of processed instances: "
						+ processingThreads[i].getNumOfProcessedInstance());
			}

			data.reset();

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")");
		}
		if (this.hiddenVectorsFile != null)
			this.writeHiddenVectors(this.hiddenVectorsFile);

	}

	public void writeHiddenVectors(String outfname) {
		try {
			BufferedWriter bf = new BufferedWriter(new FileWriter(outfname));

			for (int i = 0; i < this.d; i++) {
				int hi = fh.getIndex(1, i);

				bf.write(i + "," + hi);
				// labels
				for (int j = 0; j < this.hiddendim; j++) {
					// logger.info(data.y[i][j]);
					bf.write("," + this.hiddenWeights[hi].get(j));
				}

				bf.write("\n");
			}

			bf.close();
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}

	protected double[] getHiddenRepresentation(AVPair[] x) {
		double[] hiddenRepresentation = new double[hiddendim];
		// aggregate the the word2vec representation
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			int hi = fh.getIndex(1, x[i].index);
			// int sign = fh.getSign(1, x[i].index);

			for (int j = 0; j < hiddendim; j++) {
				hiddenRepresentation[j] += x[i].value * hiddenWeights[hi].get(j);
				sum += x[i].value;
			}

		}

		if (sum > 0.0000001) {
			for (int j = 0; j < hiddendim; j++) {
				hiddenRepresentation[j] /= sum;
			}
		}

		return hiddenRepresentation;
	}
	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 1.0;

		double[] hiddenRepresentation = this.getHiddenRepresentation(x);
		
		int treeIndex = this.tree.getTreeIndex(label);

		posterior *= getPartialPosteriors(hiddenRepresentation, treeIndex);

		while(treeIndex > 0) {

			treeIndex = this.tree.getParent(treeIndex); //Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(hiddenRepresentation, treeIndex);

		}
		//if(posterior > 0.5) logger.info("Posterior: " + posterior + "Label: " + label);
		return posterior;
	}

	public double getPartialPosteriors(double[] x, int label) {
		double posterior = 0.0;
		
		
		for (int i = 0; i < this.hiddendim; i++) {			
			posterior += x[i] * (1.0/this.scalararray[label].get()) * this.w[label].get(i);
		}
		
		posterior += (1.0/this.scalararray[label].get()) * this.bias[label].get(); 
		posterior = s.value(posterior);		
		
		return posterior;

	}

	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		double[] hiddenRepresentation = this.getHiddenRepresentation(x);
		
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
					queue.add(new NodePLT(childNode, currentP * getPartialPosteriors(hiddenRepresentation, childNode)));
				}
				
			} else {
				
				positiveLabels.add(new EstimatePair(this.tree.getLabelIndex(node.treeIndex), currentP));
				foundTop++;
				
			}
		}

		return positiveLabels;
	}

	public class UpdateThread implements Runnable {
		DataManager data = null;
		protected double[] updatevec = null;
		private final int ID;
		private int numOfProcessedInstance = 0;

		public int getID() {
			return ID;
		}

		public void setNumOfProcessedInstance(int numOfProcessedInstance) {
			this.numOfProcessedInstance = numOfProcessedInstance;
		}

		public int getNumOfProcessedInstance() {
			return numOfProcessedInstance;
		}

		public UpdateThread(DataManager data, int ID) {
			this.data = data;
			this.updatevec = new double[hiddendim];
			this.ID = ID;
		}

		@Override
		public void run() {
			// System.out.println("Thread " + this.ID );
			// Instance instance = null;
			// while( ( instance = data.getNextInstance() ) != null ){
			while (true) {
				Instance instance = data.getNextInstance();
				if (instance == null)
					break;
				this.numOfProcessedInstance++;
				// System.out.println("Thread " + this.ID +" Processed instance:
				// " + this.numOfProcessedInstance );

				double[] hiddenRepresentation = getHiddenRepresentation(instance.x);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				for (int j = 0; j < instance.y.length; j++) {

					int treeIndex = tree.getTreeIndex(instance.y[j]); // +
																		// traindata.m
																		// - 1;
					positiveTreeIndices.add(treeIndex);

					while (treeIndex > 0) {

						treeIndex = tree.getParent(treeIndex); // Math.floor((treeIndex
																// - 1)/2);
						positiveTreeIndices.add(treeIndex);

					}
				}

				if (positiveTreeIndices.size() == 0) {

					negativeTreeIndices.add(0);

				} else {

					for (int positiveNode : positiveTreeIndices) {

						if (!tree.isLeaf(positiveNode)) {

							for (int childNode : tree.getChildNodes(positiveNode)) {

								if (!positiveTreeIndices.contains(childNode)) {
									negativeTreeIndices.add(childNode);
								}

							}

						}

					}
				}

				Arrays.fill(this.updatevec, 0.0);

				// logger.info("Negative tree indices: " +
				// negativeTreeIndices.toString());
				for (int j : positiveTreeIndices) {

					double posterior = getPartialPosteriors(hiddenRepresentation, j);
					double inc = -(1.0 - posterior);

					updatedTreePosteriors(hiddenRepresentation, j, inc);
				}

				for (int j : negativeTreeIndices) {

					if (j >= t)
						logger.info("ALARM");

					double posterior = getPartialPosteriors(hiddenRepresentation, j);
					double inc = -(0.0 - posterior);

					updatedTreePosteriors(hiddenRepresentation, j, inc);
				}

				updateHiddenRepresentation(instance);

			}

		}

		synchronized protected void updateHiddenRepresentation(Instance instance) {

			double sum = 0.0;
			for (int j = 0; j < instance.x.length; j++) {
				sum += instance.x[j].value;
			}
			if (sum > 0.000001) {
				sum = 1.0 / sum;
			}

			for (int i = 0; i < instance.x.length; i++) {
				int hi = fh.getIndex(1, instance.x[i].index);
				// int sign = fh.getSign(1, instance.x[i].index);

				double learningRate = gamma / (1 + gamma * lambda * Tarrayhidden[hi].get());
				Tarrayhidden[hi].addAndGet(1);
				scalararrayhidden[hi].set(scalararrayhidden[hi].get() * (1 + learningRate * lambda));

				for (int j = 0; j < hiddendim; j++) {
					// double gradient = this.scalararrayhidden[hi] * inc *
					// this.w[ind][j] * sum * instance.x[i].value;
					double gradient = scalararrayhidden[hi].get() * this.updatevec[j] * sum * instance.x[i].value;
					double update = (learningRate * gradient);// / this.scalar;
					hiddenWeights[hi].addAndGet(j, -update);
				}

			}
		}

		synchronized protected void updatedTreePosteriors(double[] x, int label, double inc) {
			double learningRate = gamma / (1 + gamma * lambda * Tarray[label].get());
			Tarray[label].addAndGet(1);
			scalararray[label].set(scalararray[label].get() * (1 + learningRate * lambda));

			int n = x.length;

			for (int i = 0; i < hiddendim; i++) {
				double gradient = scalararray[label].get() * inc * x[i];
				double update = (learningRate * gradient);// / this.scalar;
				w[label].getAndAdd(i, -update);

				this.updatevec[i] += inc * w[label].get(i);
			}

			double gradient = scalararray[label].get() * inc;
			double update = (learningRate * gradient);// / this.scalar;
			bias[label].addAndGet(-update);
		}

		public double getPartialPosteriors(double[] x, int label) {
			double posterior = 0.0;

			for (int i = 0; i < hiddendim; i++) {
				posterior += x[i] * w[label].get(i);
			}

			posterior += bias[label].get();
			posterior = s.value(posterior);

			return posterior;

		}

	}	
	
}
