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

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class PLTFH extends MLLRFH {
	
	protected int t = 0;
	//protected double innerThreshold = 0.15;

	//protected double[] scalars = null;
	
	public PLTFH(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		System.out.println("#####################################################" );
		System.out.println("#### Leraner: PLTFT" );

		//this.innerThreshold = Double.parseDouble(this.properties.getProperty("IThreshold", "0.15") );
		//System.out.println("#### Inner node threshold : " + this.innerThreshold );
		//System.out.println("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		this.t = 2 * this.m - 1;

		int seed = 1;
		//this.hd = 50000000;//40000000;

		this.fh = new MurmurHasher(seed, this.hd, this.t);

		System.out.println( "Num. of labels: " + this.m + " Dim: " + this.d );
		System.out.println( "Num. of inner node of the trees: " + this.t  );

		System.out.print( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		//this.scalars = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}
		
		//how to initialize w?
		
		System.out.println( "Done." );
	}

		
	@Override
	public void train(AVTable data) {
		
		this.T = 1;
				
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = new ArrayList<Integer>();
			for (int i = 0; i < this.traindata.n; i++) {
				indirectIdx.add(new Integer(i));
			}

			Collections.shuffle(indirectIdx);

			for (int i = 0; i < traindata.n; i++) {

				//this.learningRate = 0.5 / (Math.ceil(this.T / ((double) this.step)));
				this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.T);
				//this.scalar *= (1 - this.learningRate * this.lambda);
				this.scalar *= (1 + this.learningRate * this.lambda);
				
				int currIdx = indirectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				for (int j = 0; j < traindata.y[currIdx].length; j++) {

					int treeIndex = traindata.y[currIdx][j] + traindata.m - 1;
					positiveTreeIndices.add(treeIndex);

					while(treeIndex > 0) {

						treeIndex = (int) Math.floor((treeIndex - 1)/2);
						positiveTreeIndices.add(treeIndex);

					}
				}

				if(positiveTreeIndices.size() == 0) {

					negativeTreeIndices.add(0);

				} else {

					PriorityQueue<Integer> queue = new PriorityQueue<Integer>();
					queue.add(0);

					while(!queue.isEmpty()) {

						int node = queue.poll();
						int leftchild = 2 * node + 1;
						int rightchild = 2 * node + 2;

						Boolean left = false, right = false;

						if(positiveTreeIndices.contains(leftchild)) {
							queue.add(leftchild);
							left = true;
						}

						if(positiveTreeIndices.contains(rightchild)) {
							queue.add(rightchild);
							right = true;
						}

						if(left == true && right == false) {
							negativeTreeIndices.add(rightchild);
						}

						if(left == false && right == true) {
							negativeTreeIndices.add(leftchild);
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

				if ((i % 50000) == 0) {
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

		int treeIndex = label + this.m - 1;

		posterior *= getPartialPosteriors(x, treeIndex);

		while(treeIndex > 0) {

			treeIndex = (int) Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(x, treeIndex);

		}
		//if(posterior > 0.5) System.out.println("Posterior: " + posterior + "Label: " + label);
		return posterior;
	}

	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		class Node {

			int treeIndex;
			double p;

			Node(int treeIndex, double p) {
				this.treeIndex = treeIndex;
				this.p = p;
			}

			@Override
			public String toString() {
				return new String("(" + this.treeIndex + ", " + this.p + ")");
			}
		};

		class NodeComparator implements Comparator<Node> {
	        @Override
			public int compare(Node n1, Node n2) {
	        	return (n1.p > n2.p) ? 1 : -1;
	        }
	    } ;

	    NodeComparator nodeComparator = new NodeComparator();

		PriorityQueue<Node> queue = new PriorityQueue<Node>(11, nodeComparator);

		queue.add(new Node(0,1.0));

		while(!queue.isEmpty()) {

			Node node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if(currentP > this.thresholds[node.treeIndex]) {

				if(node.treeIndex < this.m - 1) {
					int leftchild = 2 * node.treeIndex + 1;
					int rightchild = 2 * node.treeIndex + 2;

					queue.add(new Node(leftchild, currentP));
					queue.add(new Node(rightchild, currentP));

				} else {

					positiveLabels.add(node.treeIndex - this.m + 1);

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	
	@Override
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

		class Node {

			int treeIndex;
			double p;

			Node(int treeIndex, double p) {
				this.treeIndex = treeIndex;
				this.p = p;
			}

			@Override
			public String toString() {
				return new String("(" + this.treeIndex + ", " + this.p + ")");
			}
		};

		class NodeComparator implements Comparator<Node> {
	        @Override
			public int compare(Node n1, Node n2) {
	        	return (n1.p > n2.p) ? 1 : -1;
	        }
	    } ;

	    NodeComparator nodeComparator = new NodeComparator();

		PriorityQueue<Node> queue = new PriorityQueue<Node>(11, nodeComparator);

		queue.add(new Node(0,1.0));

		while(!queue.isEmpty()) {

			Node node = queue.poll();

			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);

			if(currentP > this.thresholds[node.treeIndex]) {

				if(node.treeIndex < this.m - 1) {
					int leftchild = 2 * node.treeIndex + 1;
					int rightchild = 2 * node.treeIndex + 2;

					queue.add(new Node(leftchild, currentP));
					queue.add(new Node(rightchild, currentP));

				} else {

					positiveLabels.add(new ComparablePair( node.treeIndex - this.m + 1, currentP ) );

				}
			}
		}

		//System.out.println("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}
	
	
	
	public void setThresholds(double[] t) {
		
		for(int j = 0; j < t.length; j++) {
			this.thresholds[j + this.m - 1] = t[j];
		}
		
		for(int j = this.m - 2; j >= 0; j--) {
			this.thresholds[j] = Math.min(this.thresholds[2*j+1], this.thresholds[2*j+2]);
		}
	}

	@Override
	public void tuneThreshold( ThresholdTuning t, AVTable data ){
		this.setThresholds(t.validate(data, this));
	}

	
	@Override
	public void loadmodel(String fname) {
		super.loadmodel(fname);
		
		//WRONG!!!
		//this.t = (this.w.length-1)/2;
	}


}
