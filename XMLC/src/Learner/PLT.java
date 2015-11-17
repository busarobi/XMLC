package Learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import preprocessing.FeatureHasher;
import threshold.TTEum;
import threshold.TTExu;
import threshold.TTOfo;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class PLT extends MLLogisticRegression {
	protected int t = 0;
	protected double innerThreshold = 0.15;
	
	public PLT(String propertyFileName) {
		super(propertyFileName);

		System.out.println("#####################################################" );
		System.out.println("#### Leraner: PLT" );
		
		this.innerThreshold = Double.parseDouble(this.properties.getProperty("IThreshold", "0.15") );
		
		System.out.println("#####################################################" );				
	}

	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		this.t = 2 * this.m - 1;
		
		System.out.println( "Num. of labels: " + this.m + " Dim: " + this.d );
		System.out.println( "Num. of inner node of the trees: " + this.t  );
		
		Random allocationRand = MasterSeed.nextRandom();
		
		System.out.print( "Allocate the learners..." );
		
		this.w = new double[this.t][];
		this.bias = new double[this.t];


		for (int i = 0; i < this.t; i++) {
			this.w[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * allocationRand.nextDouble() - 1.0;

			this.bias[i] = 2.0 * allocationRand.nextDouble() - 1.0;

//			if ((i % 100) == 0)
//				System.out.println( "Model: "+ i +" (" + this.m + ")" );

		}

		if (this.geomWeighting) {
			this.grad = new double[this.t][];
			this.gradbias = new double[this.t];
			
			for (int i = 0; i < this.t; i++) {				
				this.grad[i] = new double[d];				
			}			
		}
		
		this.thresholds = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.2;
		}
		System.out.println( "Done." );
		
	}
	
	@Override
	public void train(AVTable data) {
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
					
					updatedPosteriors( currIdx, j, mult, inc );										
				}
					
				for(int j:negativeTreeIndices) {
					
					if(j >= this.t) System.out.println("ALARM");
					
					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = (0.0 - posterior);
					
					updatedPosteriors( currIdx, j, mult, inc );						
				}
				
				this.T++;

				if ((this.T % 10000) == 0)
					System.out.println("--> Mult: " + (this.gamma * mult));
				
			}
			
		}
		
	}
	
	
	public double getPartialPosteriors(AVPair[] x, int label) {		
		double posterior = super.getPosteriors(x, label);		
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
			
			if(currentP > innerThreshold) {
			
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
	
	
	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// create the classifier and set the configuration
		AbstractLearner learner = new PLT(args[0]);

		// feature hasher
		FeatureHasher fh = null;
		
		// reading train data
		DataReader datareader = new DataReader(learner.getProperties().getProperty("TrainFile"));
		AVTable data = datareader.read();		

		if (learner.getProperties().containsKey("FeatureHashing")) {
			int featureNum = Integer.parseInt(learner.getProperties().getProperty("FeatureHashing"));
			fh = new FeatureHasher(0, featureNum);
			
			System.out.print( "Feature hashing (dim: " + featureNum + ")...");			
			data = fh.transformSparse(data);			
			System.out.println( "Done.");
		}
		
		if (learner.getProperties().containsKey("seed")) {
			long seed = Long.parseLong(learner.getProperties().getProperty("seed"));
			MasterSeed.setSeed(seed);
		}
		
		
		
		// train
		String inputmodelFile = learner.getProperties().getProperty("InputModelFile");
		if (inputmodelFile == null ) {
			learner.allocateClassifiers(data);
			learner.train(data);

			String modelFile = learner.getProperties().getProperty("ModelFile");
			learner.savemodel(modelFile);
		} else {
			learner.loadmodel(inputmodelFile);
		}
		
		
		// test
		DataReader testdatareader = new DataReader(learner.getProperties().getProperty("TestFile"));
		AVTable testdata = testdatareader.read();		
		if (fh != null ) {
			testdata = fh.transformSparse(testdata);			
		}
		
		String validFileName = learner.getProperties().getProperty("ValidFile");
		AVTable validdata = null;
		if (validFileName == null ) {
			validdata = data;
		} else {
			DataReader validdatareader = new DataReader(learner.getProperties().getProperty("ValidFile"));
			validdata = validdatareader.read();
			if (fh != null ) {
				validdata = fh.transformSparse(validdata);			
			}
			
		}
		
		
		// evaluate (EUM)
		ThresholdTuning th = new TTEum( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);
        		
		// evaluate (OFO)
		th = new TTOfo( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perfOFO = Evaluator.computePerformanceMetrics(learner, testdata);

		// evaluate (EXU)
		th = new TTExu( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perfEXU = Evaluator.computePerformanceMetrics(learner, testdata);


		for ( String perfName : perf.keySet() ) {
			System.out.println("##### EUM" + perfName + ": "  + perf.get(perfName));
		}
		
		
		for ( String perfName : perfOFO.keySet() ) {
			System.out.println("##### OFO" + perfName + ": "  + perfOFO.get(perfName));
		}

		
		for ( String perfName : perfEXU.keySet() ) {
			System.out.println("##### EXU " + perfName + ": "  + perfEXU.get(perfName));
		}


	}
	
	@Override
	public void loadmodel(String fname) {		
		super.loadmodel(fname);
		this.t = (this.w.length-1)/2;
	}
		

}
