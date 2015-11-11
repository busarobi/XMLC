package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Map;
import java.util.Random;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import preprocessing.FeatureHasher;
import threshold.TTEum;
import threshold.TTExu;
import threshold.TTOfo;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class MLLogisticRegressionNSampling extends MLLogisticRegression {
	
	// uniform = 0
	// uniform sampling of negatives, the number of negatives to be updated is negativeSamplingParameter
	// times more than as many positives are
		
	protected int negativeSamplingMode = 0;
	protected double negativeSamplingParameter = 1.0;
	
	public MLLogisticRegressionNSampling(String propertyFileName) {
		super(propertyFileName);
		System.out.println("#####################################################" );
		System.out.println("#### Leraner: LogReg with Negative Sampling" );

		// negative sampling mode
		this.negativeSamplingMode = Integer.parseInt(this.properties.getProperty("nsamp", "0") );
		this.negativeSamplingParameter = Double.parseDouble(this.properties.getProperty("nsamppar", "1.0") );
		if (this.negativeSamplingMode == 0 ) {
			System.out.println("#### Negative sampling : uniform");
			System.out.format("#### The number of negatives to be updated is %1.1f x the number of positives\n", this.negativeSamplingParameter);
		}
		
		
		System.out.println("#####################################################" );		
		
	}
	

	public void train( AVTable data ){
		this.T = 1;
		Random rand = MasterSeed.nextRandom();
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			
			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>(this.traindata.n);
			for (int i = 0; i < this.traindata.n; i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx,shuffleRand);
			

			for (int i = 0; i < traindata.n; i++) {
				double mult = 1.0 / (Math.ceil(this.T / ((double) this.step)));
				int currIdx = indiriectIdx.get(i);

				
				
				// select all positive to update
				ArrayList<Integer> indicesToUpdate = new ArrayList<Integer>();
				ArrayList<Double> indicesToUpdateLabel = new ArrayList<Double>();
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					indicesToUpdate.add(traindata.y[currIdx][j]);
					indicesToUpdateLabel.add(1.0);
					
					//labelDistribution[traindata.y[currIdx][j]]++;
				}

				// select negatives to update
				if (this.negativeSamplingMode == 0) {

					// Tricky way to select numOfNegatives number of negatives
					// randomly out of all negatives
					int indexy = 0;
					int numOfNegatives = traindata.m - traindata.y[currIdx].length;
					int numToSelect = (int) Math.ceil(this.negativeSamplingParameter * traindata.y[currIdx].length);
					
					if (numToSelect>numOfNegatives) {
						numToSelect = numOfNegatives;
					}
					
					
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
				
					double inc = (currLabel - posterior);
					updatedPosteriors( currIdx, currLabelPos, mult, inc );
				}
				
				this.T++;

				if ((i % 10000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					System.out.println("  --> Mult: " + (this.gamma * mult));
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					System.out.println("Weight: " + this.w[0][0] );
				}

			}

			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			
			
			//save model !!!!!!!!!!!!!!!!!!!!!!!
			//String modelFile = this.getProperties().getProperty("ModelFile");
			//this.savemodel(modelFile);
		}

	}
	

	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// create the classifier and set the configuration
		MLLogisticRegressionNSampling learner = new MLLogisticRegressionNSampling(args[0]);

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

}
