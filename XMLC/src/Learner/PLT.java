package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.DenseVectorExt;
import Learner.step.StepFunction;
import util.MasterSeed;

public class PLT extends MLLogisticRegression {
	private static final long serialVersionUID = -7781940205438325371L;

	private static Logger logger = LoggerFactory.getLogger(PLT.class);

	protected int t = 0;
	protected double innerThreshold = 0.15;

	public PLT(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Leraner: PLT" );

		this.innerThreshold = Double.parseDouble(this.properties.getProperty("IThreshold", "0.15") );
		logger.info("#### Inner node threshold : " + this.innerThreshold );
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		this.t = 2 * this.m - 1;

		logger.info( "Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info( "Num. of inner node of the trees: " + this.t  );

		Random allocationRand = MasterSeed.nextRandom();

		logger.info( "Allocate the learners..." );

		this.w = new DenseVectorExt[this.t];
		this.stepfunctions = new StepFunction[this.t];
		for (int i = 0; i < this.t; i++) {
			this.w[i] = new DenseVectorExt(d + 1);
			this.stepfunctions[i] = this.stepFunction.clone();

			for (int j = 0; j <= d; j++)
				this.w[i].set(j, 2.0 * allocationRand.nextDouble() - 1.0);
		}

		this.thresholds = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.2;
		}
		logger.info( "Done." );

	}

	@Override
	public void train(AVTable data) {
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>();
			for (int i = 0; i < this.traindata.n; i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx);

			for (int i = 0; i < traindata.n; i++) {
				int currIdx = indiriectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				//logger.info("Positive Labels: ");

				for (int j = 0; j < traindata.y[currIdx].length; j++) {

					//if(j == traindata.y[currIdx].length - 1)
					//	logger.info(traindata.y[currIdx][j]);
					//else
					//	logger.info(traindata.y[currIdx][j] + ", ");

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

					//logger.info("Positive tree indices: ");

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
						//	logger.info(node);
						} else {
						//	logger.info(node + ", ");
						}

					}
				}

				//logger.info("Negative tree indices: " + negativeTreeIndices.toString());


				for(int j:positiveTreeIndices) {

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = posterior - 1.0;

					updatedPosteriors( currIdx, j, inc );
				}

				for(int j:negativeTreeIndices) {

					if(j >= this.t) logger.info("ALARM");

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = posterior - 0.0;

					updatedPosteriors( currIdx, j, inc );
				}

				this.T++;

				if ((i % 10000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					logger.info("\t\t" + this.stepfunctions[0].toString() );
					logger.info("\t\tWeight: " + this.w[0].get(0) );
				}

			}

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
		}

	}


	public double getPartialPosteriors(AVPair[] x, int label) {
		double posterior = super.getPosteriors(x, label);
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

		//logger.info("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}


	public void load(String fname) {
		super.load(fname);
		this.t = (this.w.length-1)/2;
	}


}
