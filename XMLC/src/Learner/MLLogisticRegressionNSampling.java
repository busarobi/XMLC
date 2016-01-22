package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Learner.step.StepFunction;
import util.MasterSeed;

public class MLLogisticRegressionNSampling extends MLLogisticRegression {
	private static final long serialVersionUID = -3088978460592180002L;


	private static Logger logger = LoggerFactory.getLogger(MLLogisticRegressionNSampling.class);


	// uniform = 0
	// uniform sampling of negatives, the number of negatives to be updated is negativeSamplingParameter
	// times more than as many positives are

	protected int negativeSamplingMode = 0;
	protected double negativeSamplingParameter = 1.0;

	protected int[][] labelDistribution = null;
	protected int[][] truelabelDistribution = null;

	protected double[] P = null;
	protected double[] Pprime = null;


	public MLLogisticRegressionNSampling(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		logger.info("#####################################################" );
		logger.info("#### Leraner: LogReg with Negative Sampling" );

		// negative sampling mode
		this.negativeSamplingMode = Integer.parseInt(this.properties.getProperty("nsamp", "0") );
		this.negativeSamplingParameter = Double.parseDouble(this.properties.getProperty("nsamppar", "1.0") );
		if (this.negativeSamplingMode == 0 ) {
			logger.info("#### Negative sampling : uniform");
			System.out.format("#### The number of negatives to be updated is %1.1f x the number of positives\n", this.negativeSamplingParameter);
		}


		logger.info("#####################################################" );

	}

	@Override
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);

		this.labelDistribution = new int[this.m][];
		this.truelabelDistribution = new int[this.m][];

		for( int i = 0; i < this.m; i++){
			this.labelDistribution[i] = new int[2];
			this.truelabelDistribution[i] = new int[2];
		}

		this.P = new double[this.m];
		this.Pprime = new double[this.m];

		for( int i = 0; i < this.m; i++){
			this.P[i] = 1.0;
			this.Pprime[i] = 1.0;
		}

	}

	@Override
	public void train( AVTable data ){
		this.T = 1;
		Random rand = MasterSeed.nextRandom();

		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>(this.traindata.n);
			for (int i = 0; i < this.traindata.n; i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx,shuffleRand);


			for (int i = 0; i < traindata.n; i++) {
				int currIdx = indiriectIdx.get(i);



				// select all positive to update
				ArrayList<Integer> indicesToUpdate = new ArrayList<Integer>();
				ArrayList<Double> indicesToUpdateLabel = new ArrayList<Double>();
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					indicesToUpdate.add(traindata.y[currIdx][j]);
					indicesToUpdateLabel.add(1.0);

					this.truelabelDistribution[traindata.y[currIdx][j]][0]++;
					this.labelDistribution[traindata.y[currIdx][j]][0]++;
				}

				//double upweight = 1.0;
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

					//upweight = ((double)numToSelect/(double)numOfNegatives);

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

							this.labelDistribution[j][1]++;
						}
						this.truelabelDistribution[j][1]++;
					}

				}

				// update weights

				for (int j = 0; j < indicesToUpdate.size(); j++) {
					int currLabelPos = indicesToUpdate.get(j);
					double currLabel = indicesToUpdateLabel.get(j);

					double posterior = getPosteriors(traindata.x[currIdx],
							currLabelPos);

					double inc = posterior - currLabel;
					updatedPosteriors( currIdx, currLabelPos, inc );
//					if (currLabel>0.0)
//						updatedPosteriors( currIdx, currLabelPos, mult, inc );
//					else
//						updatedPosteriors( currIdx, currLabelPos, upweight * mult, inc );
				}

				this.T++;

				if ((i % 10000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					logger.info("Weight: " + this.w[0].get(0) );
				}

			}

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			for( int i = 0; i < this.m; i++ ){
				this.P[i] = (this.truelabelDistribution[i][0])/(((double)this.truelabelDistribution[i][0]+this.truelabelDistribution[i][1]));
				this.Pprime[i] = (this.labelDistribution[i][0])/(((double)this.labelDistribution[i][0]+this.labelDistribution[i][1]));
			}

			//save model !!!!!!!!!!!!!!!!!!!!!!!
			//String modelFile = this.getProperties().getProperty("ModelFile");
			//this.savemodel(modelFile);

//			double sum = 0.0;
//			for( int i = 0; i < this.m; i++ )
//				this.labelDistribution[i] = 1 - (this.labelDistribution[i] / ((double) (this.T-1)));
//			for( int i = 0; i < this.m; i++ )
//				sum += this.labelDistribution[i];
//			for( int i = 0; i < this.m; i++ )
//				this.labelDistribution[i] /= sum;
//			this.truelabelDistribution = AVTable.getPrior( data );
		}

	}


	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double post = super.getPosteriors(x, label);
		if (this.truelabelDistribution!=null)
			post = post * ( this.Pprime[label] / this.P[label] );
		return post;
	}


}
