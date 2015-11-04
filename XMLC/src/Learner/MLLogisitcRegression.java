package Learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import IO.DataReader;
import IO.Evaluator;
import IO.Result;

public class MLLogisitcRegression extends AbstractLearner {
	protected int epochs = 20;

	protected double[][] w = null;
	protected double[][] grad = null;
	protected double[] bias = null;
	protected double[] gradbias = null;

	protected double gamma = 0.6; // learning rate
	protected int step = 200;
	protected double delta = 0.1;

	// uniform sampling of negatives, the number of negatives is r times more as
	// many as positives
	protected int r = 1;

	protected int T = 1;	
	protected AVTable traindata = null;

	// 0 = "vanila"
	protected int updateMode = 0;

	protected Random rand = new Random();

	public MLLogisitcRegression(String propertyFileName) {
		super(propertyFileName);
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

		this.w = new double[this.m][];
		this.bias = new double[this.m];

		this.grad = new double[this.m][];
		this.gradbias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.w[i] = new double[d];
			this.grad[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * rand.nextDouble() - 1.0;

			this.bias[i] = 2.0 * rand.nextDouble() - 1.0;

		}

		this.thresholds = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.2;
		}
		
		// learning rate
		this.gamma = 3.0;
		// step size for learning rate
		this.step = 20000;
		this.T = 1;
		// decay of gradient
		this.delta = 0.9;
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

				int indexy = 0;
				for (int j = 0; j < traindata.m; j++) {
					double posterior = getPosteriors(traindata.x[currIdx], j);

					double currLabel = 0.0;
					if ((indexy < traindata.y[currIdx].length) && (traindata.y[currIdx][indexy] == j)) {
						currLabel = 1.0;
						indexy++;
					}

					// update the models
					double inc = (currLabel - posterior);

					int indexx = 0;
					for (int l = 0; l < this.d; l++) {
						if ((indexx < traindata.x[currIdx].length) && (traindata.x[currIdx][indexx].index == l)) {
							this.grad[j][l] = (inc * traindata.x[currIdx][indexx].value) + (this.grad[j][l] * this.delta);
							indexx++;
						} else {
							this.grad[j][l] += (this.grad[j][l] * this.delta);
						}
					}

					this.gradbias[j] = inc + this.gradbias[j] * this.delta;

					//indexx = 0;
					for (int l = 0; l < this.d; l++) {
						this.w[j][l] += (this.gamma * mult * this.grad[j][l]);
					}

					this.bias[j] += (this.gamma * mult * this.gradbias[j]);

				}

				this.T++;
				
				if ((this.T % 10000) == 0)
					System.out.println("  --> Mult: " + (this.gamma * mult));


			}

			System.out.println("--> Epoch: " +  ep + " (" + this.epochs +")" );
		}

	}

	
	public void validateThresholdEUM( AVTable data ) {
		
		double avgFmeasure = 0.0;
		// for labels
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		for( int i = 0; i < this.m; i++ ) {
			ArrayList<ComparablePair> posteriors = new ArrayList<>();
			int[] labels = new int[data.n];
			
			for( int j = 0; j < data.n; j++ ) {
				double post = this.getPosteriors(data.x[j], i);
				//System.out.println ( post );
				ComparablePair entry = new ComparablePair( post, j);
				posteriors.add(entry);
			}
			
			Collections.sort(posteriors);
			
			// assume that the labels are ordered
			int numOfPositives = 0;
			for( int j = 0; j < data.n; j++ ) {
				if ( (indices[j] < data.y[j].length) &&  (data.y[j][indices[j]] == i) ){ 
					labels[j] = 1;
				    numOfPositives++;
				    indices[j]++;
				} else {
					labels[j] = 0;
//					if ((indices[j] < data.y[j].length) && (data.y[j][indices[j]] < j ) ) 
//						indices[j]++;
				}
			}

			// tune the threshold			
			// every instance is predicted as positive first with a threshold = 0.0
			int tp = numOfPositives;
			int predictedPositives = data.n;
			double Fmeasure = ((double) tp) / ((double) ( numOfPositives + predictedPositives )); 
			double maxthreshold = 0.0;
			double maxFmeasure = Fmeasure;

			
			for( int j = 0; j < data.n; j++ ) {				
				int ind = posteriors.get(j).getValue();
				if ( labels[ind] == 1 ){
					tp--;
				}
				predictedPositives--;
				
				Fmeasure = ((double) tp) / ((double) ( numOfPositives + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = posteriors.get(j).getKey();
				}
			}			
			
			System.out.println( "Class: " + i +" (" + numOfPositives + ")\t" 
			                         +" F: " + String.format("%.4f", maxFmeasure ) 
			                         + " Th: " + String.format("%.4f", maxthreshold) );
			
			this.thresholds[i] = maxthreshold;
			avgFmeasure += maxFmeasure;
			
		}
		
		System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) this.m) ) ;
		
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

	@Override
	public Evaluator test(AVTable data) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// create the classifier and set the configuration
		MLLogisitcRegression learner = new MLLogisitcRegression(args[0]);

		// reading train data
		DataReader datareader = new DataReader(learner.getProperties().getProperty("TrainFile"));
		AVTable data = datareader.read();

		// train
		learner.allocateClassifiers(data);
		learner.train(data);

		// validate threshold
		learner.validateThresholdEUM(data);
		
		// test
		DataReader testdatareader = new DataReader(learner.getProperties().getProperty("TestFile"));
		AVTable testdata = testdatareader.read();

		// evaluate
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);
		for ( String perfName : perf.keySet() ) {
			System.out.println("##### " + perfName + ": "  + perf.get(perfName));
		}
		
	}

}
