package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Learner.step.StepFunction;
import util.MasterSeed;

public class MLLRFHR extends MLLRFH {
	private static final long serialVersionUID = 867786178489831896L;


	private static Logger logger = LoggerFactory.getLogger(MLLRFHR.class);


	protected int[] Tarray = null;	
	protected double[] scalararray = null;

	
	public MLLRFHR(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		shuffleRand = MasterSeed.nextRandom();		
		
		logger.info("#####################################################" );
		logger.info("#### Leraner: MLLRFH" );

		
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);
		
		
		this.Tarray = new int[this.m];
		this.scalararray = new double[this.m];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
				
		//how to initialize w?
		
		logger.info( "Done." );
	}
	
	

	protected void updatedPosteriors( int currIdx, int label, double inc) {
	
		
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		//logger.info(this.learningRate + "\t" + this.scalar[label]);
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			//logger.info(sign);
			//double gradient = inc * traindata.x[currIdx][i].value; 
			//double update = this.learningRate * gradient;
			//this.w[index] -= update; 
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
			//logger.info("w -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
			
		}
		
		
		// Include bias term in weight vector:
		//int biasIndex = fh.getIndex(label, -1);
		//double gradient = inc;
		//double update = this.learningRate * gradient;	
		//this.w[biasIndex] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);

		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
	}

	
	

	@Override
	public void train(AVTable data) {
		
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			ArrayList<Integer> indirectIdx = this.shuffleIndex();

			for (int i = 0; i < traindata.n; i++) {
								
				int currIdx = indirectIdx.get(i);

				int indexy = 0;
				for (int label = 0; label < traindata.m; label++) {										
					double posterior = getPosteriors(traindata.x[currIdx], label);

					double currLabel = 0.0;
					
					if ((indexy < traindata.y[currIdx].length) && (traindata.y[currIdx][indexy] == label)) {
						currLabel = 1.0;
						indexy++;
					}

					// update the models
					double inc = posterior - currLabel;

					updatedPosteriors( currIdx, label, inc);
										
				}

				

				if ((i % 10000) == 0) {
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




}
