package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Properties;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;

import Data.AVPair;
import Data.AVTable;
import Data.SparseVectorExt;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import preprocessing.FeatureHasher;
import preprocessing.FeatureHasherFactory;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;
import util.HashFunction;
import util.MasterSeed;

public class MLLRFHRNS extends MLLRFHR {

	int [] numOfUpdates = null; 
	int [] numOfPositiveUpdates = null;
	int [] numOfNegativeUpdates = null;
	double [] contextChange = null;
	
	int samplingRatio = 1;
	
	public MLLRFHRNS(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		
		shuffleRand = MasterSeed.nextRandom();
		
		System.out.println("#####################################################" );
		System.out.println("#### Learner: MLLRFHRNS" );
		
		// sampling ratio 
		this.samplingRatio = Integer.parseInt(this.properties.getProperty("samplingRate", "1"));
		System.out.println("#### Sampling ratio: " + this.samplingRatio );
		
		System.out.println("#####################################################" );
		
	}

	@Override
	public void allocateClassifiers(AVTable data) {
				
		super.allocateClassifiers(data);
		
		this.numOfUpdates = new int[this.m]; 
		this.numOfPositiveUpdates = new int[this.m];
		this.numOfNegativeUpdates = new int[this.m];
		
		this.contextChange = new double[this.m];
		
		
		System.out.println( "Done." );
	}
	
	

	protected void updatedPosteriors( int currIdx, int label, double inc) {
	
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);

		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);		
			this.w[index] -= update; 
		}

		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);		
		this.bias[label] -= update;

	}

	protected ArrayList<Integer> shuffleIndex() {
		ArrayList<Integer> indirectIdx = new ArrayList<Integer>(this.traindata.n);
		for (int i = 0; i < this.traindata.n; i++) {
			indirectIdx.add(new Integer(i));
		}
		Collections.shuffle(indirectIdx, shuffleRand);
		return indirectIdx;
	}

	@Override
	public void train(AVTable data) {
		
		HashSet<Integer> positiveLabels = new HashSet<>(); 
		HashSet<Integer> negativeLabels = new HashSet<>();
		
		Random random = new Random(1);
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();

			for (int i = 0; i < traindata.n; i++) {
				
				int currIdx = indirectIdx.get(i);

				int numOfNegToSample = traindata.y[currIdx].length * this.samplingRatio;
				
				positiveLabels.clear();
				negativeLabels.clear();
				
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					int label = traindata.y[currIdx][j];
					positiveLabels.add(label);
					numOfUpdates[label]++;
					numOfPositiveUpdates[label]++;
				}
				
				while(negativeLabels.size() < numOfNegToSample && negativeLabels.size() < this.m - numOfNegToSample ) {
					
					int label = random.nextInt(this.m);
					
					if(!positiveLabels.contains(label)) {
					
						negativeLabels.add(label);
						numOfUpdates[label]++;
						numOfNegativeUpdates[label]++;
					}
				}
				
				//System.out.println("Positive labels: " + positiveLabels.toString());
				
				//System.out.println("Negative labels: " + negativeLabels.toString());
				
				for(int j:positiveLabels) {

					double posterior = getUncalibratedPosteriors(traindata.x[currIdx],j);
					double inc = -(1.0 - posterior); 

					updatedPosteriors(currIdx, j, inc);
				}

				for(int j:negativeLabels) {

					double posterior = getUncalibratedPosteriors(traindata.x[currIdx],j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(currIdx, j, inc);
				}	
				
				//if(i == 5) System.exit(1);
				
				if ((i % 100000) == 0) {
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

		for(int i = 0; i < this.m; i++) {
			this.contextChange[i] = this.computeContextChange(this.numOfPositiveUpdates[i], this.numOfUpdates[i], (this.traindata.n * this.epochs));
//			System.out.println(i + ": " + numOfUpdates[i] + " " + (this.epochs * this.traindata.n) + " " + (((double) this.numOfUpdates[i]/(double) (this.epochs * this.traindata.n))));
		}

	
	}

	public double computeContextChange(double positiveUpdates, double allUpdates, double allExamples) {
		double pi = positiveUpdates / allUpdates; //(double) this.numOfPositiveUpdates[label]/(double) this.numOfUpdates[label];
		double z = positiveUpdates / allExamples; // (double) this.numOfPositiveUpdates[label]/(double)  (this.epochs*this.traindata.n);
		if(pi == 1) {
			return 1;
		} else {
			return ((1 - pi) * z) / ((1-pi)*z + pi* (1-z));
		}
	}
	
	
	Sigmoid s = new Sigmoid();
	@Override
	public double getPosteriors(AVPair[] x, int label) {
	 
		double posterior = getUncalibratedPosteriors(x, label);
		//System.out.print(label + "\t" + posterior);
		posterior = (contextChange[label] * posterior) / (contextChange[label] * posterior + (1 - contextChange[label]) * (1 - posterior));
		//System.out.println(" " + posterior);
		return posterior;

	}

	public double getUncalibratedPosteriors(AVPair[] x, int label) {
		
		double posterior = 0.0;
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value *sign) * (1/this.scalararray[label]) * this.w[hi];
		}
		
		posterior += (1/this.scalararray[label]) * this.bias[label]; 
		posterior = s.value(posterior);		
		
		return posterior;

	}

	
	
	@Override
	public void savemodel(String fname) {
		// TODO Auto-generated method stub
		try{
			System.out.print( "Saving model (" + fname + ")..." );						
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));

			writer.write( "d = "+ this.d + "\n" );
			writer.write( "hd = "+ this.hd + "\n" );
			writer.write( "m = "+ this.m + "\n" );
			
			// write out weights
			writer.write( ""+ (1/this.scalar) * this.w[0]/*.get(i)*/ );
			for(int i = 1; i< this.w.length; i++ ){
				writer.write( " "+ (1/this.scalar) * this.w[i]/*.get(i)*/ );
			}
			writer.write( "\n" );

			// bias
			writer.write( ""+ (1/this.scalar) * this.bias[0]/*.get(i)*/ );
			for(int i = 1; i< this.bias.length; i++ ){
				writer.write( " "+ (1/this.scalar) * this.bias[i]/*.get(i)*/ );
			}
			writer.write( "\n" );
						
			// write out threshold
			writer.write( ""+ this.thresholds[0] );
			for(int i = 1; i< this.thresholds.length; i++ ){
				writer.write( " "+ this.thresholds[i] );
			}
			writer.write( "\n" );

			writer.close();
			
			System.out.println( "Done." );
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	@Override
	public void load(String fname) {
		//to be completed
	}

}
