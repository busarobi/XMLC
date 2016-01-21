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
import java.util.Properties;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Learner.step.StepFunction;
import preprocessing.FeatureHasher;
import preprocessing.FeatureHasherFactory;
import preprocessing.UniversalHasher;
import util.MasterSeed;

public class MLLRFH extends AbstractLearner {
	private static final long serialVersionUID = -3780864873985503188L;

	private static Logger logger = LoggerFactory.getLogger(MLLRFH.class);

	transient protected int epochs = 1;
	protected int fhseed = 1;
	protected double[] w = null;

	transient protected double gamma = 0; // learning rate
	transient protected int step = 0;
	transient Sigmoid s = new Sigmoid();


 
	transient protected int T = 1;
	transient protected AVTable traindata = null;

	Random shuffleRand;
	
	transient protected FeatureHasher fh = null;
	
	protected int hd;


	protected double[] bias;

	transient protected double learningRate = 1.0;
	protected double scalar = 1.0;
	transient protected double lambda = 0.00001;

	protected String hasher = "Universal";
	
	
	public MLLRFH(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		shuffleRand = MasterSeed.nextRandom();
		this.scalar = 1.0;
		
		logger.info("#####################################################" );
		logger.info("#### Leraner: MLLRFH" );

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
		
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

				
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.m);		
		
		logger.info( "Num. of labels: " + this.m + " Dim: " + this.d + " Hash dim: " + this.hd );
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.m];
		this.bias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.5;
		}
		
		//how to initialize w?
		
		logger.info( "Done." );
	}
	
	

	protected void updatedPosteriors( int currIdx, int label, double inc) {
	
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalar * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
		}
		
		

		
		double gradient = this.scalar * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);

		
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
		this.T = 1;
		//this.scalar = 1.0;
		//this.gamma = 0.5;
		
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			ArrayList<Integer> indirectIdx = this.shuffleIndex();

			for (int i = 0; i < traindata.n; i++) {
				
				//this.learningRate = 0.5 / (Math.ceil(this.T / ((double) this.step)));
				this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.T);
				//this.scalar *= (1 - this.learningRate * this.lambda);
				this.scalar *= (1 + this.learningRate * this.lambda);
				
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

				this.T++;

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



	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value *sign) * (1/this.scalar) * this.w[hi];
		}
		
		posterior += (1/this.scalar) * this.bias[label]; 
		posterior = s.value(posterior);		
		
		return posterior;

	}

	public void save(String fname) {
		// TODO Auto-generated method stub
		try{
			logger.info( "Saving model (" + fname + ")..." );						
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
			
			logger.info( "Done." );
		} catch (IOException e) {
			logger.info(e.getMessage());
		}

	}

	public void load(String fname) {
		try {
			logger.info( "Loading model (" + fname + ")..." );
			Path p = Paths.get(fname);

			BufferedReader reader = Files.newBufferedReader(p, Charset.forName("UTF-8"));
		    String line = null;

		    // read file
		    ArrayList<String> lines = new ArrayList<String>();
		    while ((line = reader.readLine()) != null) {
		        lines.add(line);
		    }

		    reader.close();
		    
		    // d		    
		    String[] tokens = lines.get(0).split(" ");
		    this.d = Integer.parseInt(tokens[tokens.length-1]);
		    // hd 
		    tokens = lines.get(1).split(" ");
		    this.hd = Integer.parseInt(tokens[tokens.length-1]);
		    		    
		    // m
		    tokens = lines.get(2).split(" ");
		    this.m = Integer.parseInt(tokens[tokens.length-1]);

		    
		    // process lines
		    // allocate the model
		    //this.m = lines.size()-1;
		    this.w = new double[this.hd];//new DenseVector(this.hd);

		    
		    String[] values =  lines.get(3).split( " " );
		    this.w = new double[values.length];
		    for( int j=0; j < values.length; j++ ){
		    	this.w[j] = Double.parseDouble(values[j]);
		    }
		    
		    if (this.w.length != this.hd ) {
		    	System.err.println( "Num. of weights is not appropriate!");
		    	System.exit(-1);
		    }

		    
		    values =  lines.get(4).split( " " );
		    this.bias = new double[values.length];
		    for( int j=0; j < values.length; j++ ){
		    	this.bias[j] = Double.parseDouble(values[j]);
		    }
		    
//		    if (this.bias.length != this.m ) {
//		    	System.err.println( "Num. of bias weights is not appropriate!");
//		    	System.exit(-1);
//		    }

		    
		    
		    // last line for thresholds		    
		    String[] tValues =  lines.get(lines.size()-1).split( " " );
		    this.thresholds = new double[tValues.length];
	    	for( int j=0; j < tValues.length; j++ ){
	    		this.thresholds[j] = Double.parseDouble(tValues[j]);
	    	}

	    	this.fh = new UniversalHasher(fhseed, this.hd, this.m);

	    	this.scalar=1.0;
		    logger.info( "Done." );
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}

	}

}
