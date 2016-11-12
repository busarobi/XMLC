package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.Instance;
import IO.DataManager;
import preprocessing.FeatureHasher;
import preprocessing.FeatureHasherFactory;
import util.MasterSeed;

public class MLL extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(MLL.class);
	
	transient protected int epochs = 1;
	protected int fhseed = 1;
	protected double[] w = null;

	transient protected double gamma = 0; // learning rate
	transient protected int step = 0;
	transient Sigmoid s = new Sigmoid();
 
	transient protected int T = 1;
	transient protected DataManager traindata = null;
	
	
	transient protected FeatureHasher fh = null;
	
	protected int hd;


	protected double[] bias;

	transient protected double learningRate = 1.0;
	protected double scalar = 1.0;
	transient protected double lambda = 0.00001;

	protected String hasher = "Universal";
	
	int [] numOfUpdates = null; 
	int [] numOfPositiveUpdates = null;
	int [] numOfNegativeUpdates = null;
	double [] contextChange = null;
	
	int samplingRatio = 1;
	
	Random shuffleRand = null;

	protected int[] Tarray = null;	
	protected double[] scalararray = null;
	
	
	
	public MLL(Properties properties) {
		super(properties);
		
		shuffleRand = MasterSeed.nextRandom();
		
		System.out.println("#####################################################" );
		System.out.println("#### Learner: MLL" );
		// learning rate
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "1.0"));
		logger.info("#### gamma: " + this.gamma );

		// scalar
		this.lambda = Double.parseDouble(this.properties.getProperty("lambda", "1.0"));
		logger.info("#### lambda: " + this.lambda );

		// epochs
		this.epochs = Integer.parseInt(this.properties.getProperty("epochs", "30"));
		logger.info("#### epochs: " + this.epochs );

		// epochs
		this.hasher = this.properties.getProperty("hasher", "Universal");
		logger.info("#### Hasher: " + this.hasher );
		
		
		this.hd = Integer.parseInt(this.properties.getProperty("MLFeatureHashing", "50000000")); 
		logger.info("#### Num of ML hashed features: " + this.hd );
		
		// sampling ratio 
		this.samplingRatio = Integer.parseInt(this.properties.getProperty("samplingRate", "1"));
		System.out.println("#### Sampling ratio: " + this.samplingRatio );
		
		System.out.println("#####################################################" );
		
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();

				
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.m);		
		
		logger.info( "Num. of labels: " + this.m + " Dim: " + this.d + " Hash dim: " + this.hd );
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.m];
		this.bias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.5;
		}
				
		this.Tarray = new int[this.m];
		this.scalararray = new double[this.m];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);

		
		this.numOfUpdates = new int[this.m]; 
		this.numOfPositiveUpdates = new int[this.m];
		this.numOfNegativeUpdates = new int[this.m];
		
		this.contextChange = new double[this.m];
		
		
		System.out.println( "Done." );
	}
	
	

	protected void updatedPosteriors( AVPair[] x, int label, double inc) {
	
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);

		int n = x.length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, x[i].index);
			int sign = fh.getSign(label, x[i].index);
			
			double gradient = this.scalararray[label] * inc * (x[i].value * sign);
			double update = (this.learningRate * gradient);		
			this.w[index] -= update; 
		}

		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);		
		this.bias[label] -= update;

	}


	@Override
	public void train(DataManager data) {
		
		HashSet<Integer> positiveLabels = new HashSet<>(); 
		HashSet<Integer> negativeLabels = new HashSet<>();
		
		Random random = new Random(1);
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			data.reset();
			
			while( data.hasNext() == true ){
				Instance instance = data.getNextInstance();				
				this.T++;
				
				int numOfNegToSample = instance.y.length * this.samplingRatio;
				
				positiveLabels.clear();
				negativeLabels.clear();
				
				for (int j = 0; j < instance.y.length; j++) {
					int label = instance.y[j];
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

					double posterior = getUncalibratedPosteriors(instance.x,j);
					double inc = -(1.0 - posterior); 

					updatedPosteriors(instance.x, j, inc);
				}

				for(int j:negativeLabels) {

					double posterior = getUncalibratedPosteriors(instance.x,j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(instance.x, j, inc);
				}	
				
				//if(i == 5) System.exit(1);
				
				if ((T % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ this.T );
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
			this.contextChange[i] = this.computeContextChange(this.numOfPositiveUpdates[i], this.numOfUpdates[i], (this.T * this.epochs));
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
}
