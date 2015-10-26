package Learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import IO.DataReader;
import IO.Result;

public class SGDMLC {
	protected int m = 0; // num of labels
	protected int d = 0;
	protected int epochs = 10;

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
	protected String updateMode = "vanila";

	protected Random rand = new Random();

	public SGDMLC(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		Random r = new Random();

		this.w = new double[this.m][];
		this.bias = new double[this.m];

		this.grad = new double[this.m][];
		this.gradbias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.w[i] = new double[d];
			this.grad[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * r.nextDouble() - 1.0;

			this.bias[i] = 2.0 * r.nextDouble() - 1.0;

		}

		// learning rate
		this.gamma = 0.1;
		// step size for learning rate
		this.step = 2000;
		this.T = 1;
		// decay of gradient
		this.delta = 0.01;

	}

	/*
	 * Full update of SGD
	 */

	public void train() {
		
		int[] labelDistribution = new int[this.m];
		
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

				if (updateMode.compareTo("vanila") == 0) {
					int indexy = 0;
					for (int j = 0; j < traindata.m; j++) {
						double posterior = getPosteriors(traindata.x[currIdx],
								j);
						double currLabel = 0.0;
						if ((indexy < traindata.y[currIdx].length)
								&& (traindata.y[currIdx][indexy] == j)) {
							currLabel = 1.0;
							indexy++;
						}

						// update the models
						double inc = (currLabel - posterior);

						int indexx = 0;
						for (int l = 0; l < this.d; l++) {
							if ((indexx < traindata.x[currIdx].length)
									&& (traindata.x[currIdx][indexx].index == l)) {
								this.grad[j][l] += (inc
										* traindata.x[currIdx][indexx].value + this.grad[j][l]
										* this.delta);
								indexx++;
							} else {
								this.grad[j][l] += (this.grad[j][l] * this.delta);
							}
						}

						this.gradbias[j] += (inc + this.gradbias[j]
								* this.delta);

						indexx = 0;
						for (int l = 0; l < this.d; l++) {
							this.w[j][l] += (this.gamma * mult * this.grad[j][l]);
						}

						this.bias[j] += (this.gamma * mult * this.gradbias[j]);

					}
				} else {
					// select all positive to update
					ArrayList<Integer> indicesToUpdate = new ArrayList<Integer>();
					ArrayList<Double> indicesToUpdateLabel = new ArrayList<Double>();
					for (int j = 0; j < traindata.y[currIdx].length; j++) {
						indicesToUpdate.add(traindata.y[currIdx][j]);
						indicesToUpdateLabel.add(1.0);
						
						labelDistribution[traindata.y[currIdx][j]]++;
					}

					// select negatives to update
					if (updateMode.compareTo("uniform") == 0) {

						// Tricky way to select numOfNegatives number of negatives
						// randomly out of all negatives
						int indexy = 0;
						int numOfNegatives = traindata.m - traindata.y[currIdx].length;
						int numToSelect = r * traindata.y[currIdx].length;
						
						for (int j = 0; j < traindata.m; ++j) {
							// skip the positives
							if (( indexy< traindata.y[currIdx].length ) && (traindata.y[currIdx][indexy] == j) ){
								indexy++;
								continue;
							}

							int rest = numOfNegatives - j;
							double r = rand.nextDouble();

							if ((((double) (numToSelect)) / rest) > r) {
							//if (true) {	
								--numToSelect;
								
								indicesToUpdate.add(j);
								indicesToUpdateLabel.add(0.0);								
							}
						}

					}

					// update weights
					for (int j = 0; j < indicesToUpdate.size(); j++) {
						int currLabelPos = indicesToUpdate.get(j);
						double currLabel = indicesToUpdateLabel.get(j);

						double posterior = getPosteriors(traindata.x[currIdx],
								currLabelPos);

						// update the models
						double inc = (currLabel - posterior);

						if (currLabel < 0.5 ) {
							inc *= ( 1/posterior);
							//if (labelDistribution[currLabelPos]>0)
							//	inc *= ((double) (this.T)) / ((double) labelDistribution[currLabelPos] * (r+1)  ) ; 
									
							//inc *= ((double)(this.m - traindata.y[currIdx].length)) / ((double) ( traindata.y[currIdx].length * r)); 
						}
						
						int indexx = 0;
						for (int l = 0; l < this.d; l++) {
							if ((indexx < traindata.x[currIdx].length)
									&& (traindata.x[currIdx][indexx].index == l)) {
								this.grad[currLabelPos][l] += (inc * traindata.x[currIdx][indexx].value + this.grad[currLabelPos][l]
										* this.delta);
								indexx++;
							} else {
								this.grad[currLabelPos][l] += (this.grad[currLabelPos][l] * this.delta);
							}
						}

						this.gradbias[currLabelPos] += (inc + this.gradbias[currLabelPos]
								* this.delta);

						indexx = 0;
						for (int l = 0; l < this.d; l++) {
							this.w[currLabelPos][l] += (this.gamma * mult * this.grad[currLabelPos][l]);
						}

						this.bias[currLabelPos] += (this.gamma * mult * this.gradbias[currLabelPos]);

					}

				}

				this.T++;

				if ((this.T % 10) == 0)
					System.out.println("--> Mult: " + (this.gamma * mult));

			}
		}
	}

	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		for (int i = 0; i < x.length; i++) {
			posterior += (x[i].value * this.w[label][x[i].index]);
		}
		posterior += this.bias[label];
		posterior = 1.0 / (1.0 + Math.exp(-posterior));
		return posterior;
	}

	public Result test(AVTable testdata) {
		double[][] posteriors = new double[testdata.n][];
		int[][] labels = new int[testdata.n][];

		for (int i = 0; i < testdata.n; i++) {
			posteriors[i] = new double[testdata.m];
			for (int j = 0; j < this.m; j++) {
				posteriors[i][j] = this.getPosteriors(testdata.x[i], j);
			}
		}

		Result res = new Result(posteriors, testdata);
		return res;
	}

	public static void main(String[] args) throws Exception {
		// String fileName = "/Users/busarobi/work/XMLC/data/scene/scene_train";

		//String trainfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_train.svm";
		//String testfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_test.svm";

		String trainfileName = "/Users/busarobi/work/XMLC/data/mediamill/train-exp1.svm";
		String testfileName = "/Users/busarobi/work/XMLC/data/mediamill/test-exp1.svm";

		DataReader datareader = new DataReader(trainfileName);
		AVTable data = datareader.read();

		SGDMLC learner = new SGDMLC(data);
		
		learner.setUpdateMode("uniform" );
		
		learner.train();

		//learner.trainIter( 10 );

		DataReader testdatareader = new DataReader(testfileName);
		AVTable testdata = testdatareader.read();

		Result result = learner.test(testdata);

		System.out.println("Hamming loss: " + result.getHL());

	}

	public void setUpdateMode(String updateMode) {
		this.updateMode = updateMode;
	}

}
