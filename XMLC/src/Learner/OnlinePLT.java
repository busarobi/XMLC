package Learner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import IO.DataReader;
import IO.Result;
import IO.PerformanceMeasures;

public class OnlinePLT {
	protected int m = 0; // num of labels
	protected int t = 0;  // num of tree nodes (including root, internal nodes, and leaves)
	protected int d = 0;
	protected int epochs = 100;

	protected double[][] w = null;
	protected double[][] grad = null;
	protected double[] bias = null;
	protected double[] gradbias = null;

	protected double gamma = 0.33; // learning rate
	protected int step = 20;
	
	// uniform sampling of negatives, the number of negatives is r times more as many as positives
	protected int r = 1;            
	
	protected int T = 1;
	protected double delta = 0.01;
	protected AVTable traindata = null;

	protected Random rand = new Random();

	public OnlinePLT(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		this.t = 2 * this.m - 1;
		Random r = new Random();

		this.w = new double[this.t][];
		this.bias = new double[this.t];

		System.out.println(this.m + " " + (this.t)  + " " + w.length);
		
		this.grad = new double[this.t][];
		this.gradbias = new double[this.t];

		for (int i = 0; i < this.t; i++) {
			this.w[i] = new double[d];
			this.grad[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 0.0; //2.0 * r.nextDouble() - 1.0;

			this.bias[i] = 0.0; //2.0 * r.nextDouble() - 1.0;

		}

		// learning rate
		this.gamma = 0.1;

		// step size for learning rate
		this.step = 2000;
		this.T = 1;
		
		// decay of gradient
		this.delta = 0.01;

	}

	public void train() {
		
		for (int ep = 0; ep < this.epochs; ep++) {

			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>();
			for (int i = 0; i < this.traindata.n; i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx);

			for (int i = 0; i < traindata.n; i++) {
		
				double mult = 1.0 / (Math.ceil(this.T / ((double) this.step)));
				int currIdx = indiriectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();
					
				//System.out.print("Positive Labels: ");
				
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
						
					//if(j == traindata.y[currIdx].length - 1) 
					//	System.out.println(traindata.y[currIdx][j]);
					//else
					//	System.out.print(traindata.y[currIdx][j] + ", ");
					
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
					
					//System.out.print("Positive tree indices: ");
					
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
						
						if(queue.isEmpty()) {
						//	System.out.println(node);
						} else {
						//	System.out.print(node + ", ");
						}
						
					}
				}
					
				//System.out.println("Negative tree indices: " + negativeTreeIndices.toString());
				
				
				for(int j:positiveTreeIndices) {
						
					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = (1.0 - posterior);
					
					int indexx = 0;

					for (int l = 0; l < this.d; l++) {
						if ((indexx < traindata.x[currIdx].length) && (traindata.x[currIdx][indexx].index == l)) {
							this.grad[j][l] = (inc * traindata.x[currIdx][indexx].value + this.grad[j][l] * this.delta);
							indexx++;
						} else {
							this.grad[j][l] = (this.grad[j][l] * this.delta);
						}
					}

					this.gradbias[j] = (inc + this.gradbias[j] * this.delta);

					indexx = 0;
					
					for (int l = 0; l < this.d; l++) {
					
						this.w[j][l] += (this.gamma * mult * this.grad[j][l]);
					
					}
					
					this.bias[j] += (this.gamma * mult * this.gradbias[j]);
					
				}
					
				for(int j:negativeTreeIndices) {
					
					if(j >= this.t) System.out.println("ALARM");
					
					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = (0.0 - posterior);
					
					int indexx = 0;
				
					for (int l = 0; l < this.d; l++) {
						if ((indexx < traindata.x[currIdx].length) && (traindata.x[currIdx][indexx].index == l)) {
							this.grad[j][l] = (inc * traindata.x[currIdx][indexx].value + this.grad[j][l] * this.delta);
							indexx++;
						} else {
							this.grad[j][l] = (this.grad[j][l] * this.delta);
						}
					}

					this.gradbias[j] = (inc + this.gradbias[j] * this.delta);

					indexx = 0;
					
					for (int l = 0; l < this.d; l++) {
						this.w[j][l] += (this.gamma * mult * this.grad[j][l]);
					}
					
					this.bias[j] += (this.gamma * mult * this.gradbias[j]);
					
				}
				
				this.T++;

				if ((this.T % 10000) == 0)
					System.out.println("--> Mult: " + (this.gamma * mult));
				
			}
			
		}
		
		//for(int j = 0; j<w.length; j++) {
		//	System.out.println(Arrays.toString(w[j]) + ", " + bias[j]);
		//}

	}

	public double getPartialPosteriors(AVPair[] x, int label) {
		
		double posterior = 0.0;
		
		for (int i = 0; i < x.length; i++) {
			posterior += (x[i].value * this.w[label][x[i].index]);
		}
		
		posterior += this.bias[label];
		posterior = 1.0 / (1.0 + Math.exp(-posterior));
		
		return posterior;
	}


	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 1.0;
		
		
		int treeIndex = label + this.m - 1;
		
		posterior *= getPartialPosteriors(x, treeIndex);
		
		while(treeIndex > 0) {
			
			treeIndex = (int) Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(x, treeIndex);
		
		}	
				
		return posterior;
	}

	
	
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		class Node {

			int treeIndex;
			double p;
			
			Node(int treeIndex, double p) {
				this.treeIndex = treeIndex;
				this.p = p;
			}
			
			public String toString() {
				return new String("(" + this.treeIndex + ", " + this.p + ")");
			}
		};
		
		class NodeComparator implements Comparator<Node> {
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
			
			if(currentP > 0.15) {
			
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
	
	
	public Result test(AVTable testdata) {
		double[][] posteriors = new double[testdata.n][];
		
		for (int i = 0; i < testdata.n; i++) {
			posteriors[i] = new double[testdata.m];
			for (int j = 0; j < this.m; j++) {
				posteriors[i][j] = this.getPosteriors(testdata.x[i], j);
			}
		}

		Result res = new Result(posteriors, testdata);
		return res;
	}
	
	public HashSet<Integer>[] getPredictedLabels(AVTable testdata) {
		HashSet<Integer>[] predictedLabels = new HashSet[testdata.n];
				
		for (int i = 0; i < testdata.n; i++) {
			
			predictedLabels[i] = this.getPositiveLabels(testdata.x[i]);
	
		}

		return predictedLabels;
	}

	
	public static void main(String[] args) throws Exception {

//		String trainfileName = "../data/mediamill/train-exp1.svm";
//		String testfileName = "../data/mediamill/test-exp1.svm";

		String trainfileName = "/Users/busarobi/work/XMLC/data/mediamill/train-exp1.svm";
		String testfileName = "/Users/busarobi/work/XMLC/data/mediamill/test-exp1.svm";

		//String trainfileName = "../data/test/test.svm";
		//String testfileName = "../data/test/test.svm";

		
		
		DataReader datareader = new DataReader(trainfileName);
		AVTable data = datareader.read();

		OnlinePLT learner = new OnlinePLT(data);
				
		learner.train();

		DataReader testdatareader = new DataReader(testfileName);
		AVTable testdata = testdatareader.read();

		Result result = learner.test(testdata);
		System.out.println("Hamming loss: " + result.getHL());

		PerformanceMeasures pm = new PerformanceMeasures();
		System.out.println("Hamming loss: " + pm.computeHammingLoss(learner.getPredictedLabels(testdata), testdata));
		System.out.println("Macro-F: " + pm.computeMacroF(learner.getPredictedLabels(testdata), testdata));
		
	}

}
