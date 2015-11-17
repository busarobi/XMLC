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
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import IO.DataReader;
import IO.Evaluator;
import preprocessing.FeatureHasher;
import threshold.TTEum;
import threshold.TTExu;
import threshold.TTOfo;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class MLLogisticRegression extends AbstractLearner {
	protected int epochs = 20;

	protected double[][] w = null;
	protected double[][] grad = null;
	protected double[] bias = null;
	protected double[] gradbias = null;

	protected double gamma = 0; // learning rate
	protected int step = 0;
	protected double delta = 0;
    //protected int OFOepochs = 1;
	
	protected boolean geomWeighting = true;
	
	protected int T = 1;	
	protected AVTable traindata = null;

	// 0 = "vanila"
	protected int updateMode = 0;

	Random shuffleRand;

	public MLLogisticRegression(String propertyFileName) {
		super(propertyFileName);
		shuffleRand = MasterSeed.nextRandom();
		
		System.out.println("#####################################################" );
		System.out.println("#### Leraner: LogReg" );
		
		// learning rate		
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "10.0"));
		System.out.println("#### gamma: " + this.gamma );
		
		// step size for learning rate
		this.step = Integer.parseInt(this.properties.getProperty("step", "2000") );
		System.out.println("#### step: " + this.step );
		
		// decay of gradient
		this.delta = Double.parseDouble(this.properties.getProperty("delta", "0.0") );
		System.out.println("#### delta: " + this.delta );
		if (this.delta < Double.MIN_VALUE ){
			this.geomWeighting = false;
			System.out.println( "#### No geom. weighting!");
		}
		
		this.epochs = Integer.parseInt(this.properties.getProperty("epochs", "30"));
		System.out.println("#### epochs: " + this.epochs );

		System.out.println("#####################################################" );		
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

		System.out.println( "Num. of labels: " + this.m + " Dim: " + this.d );
		Random allocationRand = MasterSeed.nextRandom();
		
		System.out.print( "Allocate the learners..." );
		
		this.w = new double[this.m][];
		this.bias = new double[this.m];


		for (int i = 0; i < this.m; i++) {
			this.w[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * allocationRand.nextDouble() - 1.0;

			this.bias[i] = 2.0 * allocationRand.nextDouble() - 1.0;

//			if ((i % 100) == 0)
//				System.out.println( "Model: "+ i +" (" + this.m + ")" );

		}

		if (this.geomWeighting) {
			this.grad = new double[this.m][];
			this.gradbias = new double[this.m];
			
			for (int i = 0; i < this.m; i++) {				
				this.grad[i] = new double[d];				
			}			
		}
		
		this.thresholds = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.2;
		}
		System.out.println( "Done." );
	}

	protected void updatedPosteriors( int currIdx, int j, double mult, double inc ) {
		if (this.geomWeighting) 
		{
			// in case of geometric weighting all weights need to be updated
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
		} else {
			// vanilla update of weights
			for( int l = 0; l < traindata.x[currIdx].length; l++) {
				this.w[j][traindata.x[currIdx][l].index] += (this.gamma * mult * inc * traindata.x[currIdx][l].value);
			}
			this.bias[j] += (this.gamma * mult * inc);
		}
		
	}
	
	
	@Override
	public void train(AVTable data) {
		this.T = 1;
		
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
					
					updatedPosteriors( currIdx, j, mult, inc );
					
				}

				this.T++;

				if ((i % 1000) == 0) {
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

	
	
	Sigmoid s = new Sigmoid();
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		for (int i = 0; i < x.length; i++) {
			posterior += x[i].value * this.w[label][x[i].index];
		}
		posterior += this.bias[label];
		posterior = s.value(posterior);
		return posterior;
	}
	
//	@Override
//	public Evaluator test(AVTable data) {
//		// TODO Auto-generated method stub
//		return null;
//	}
	
	
	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// create the classifier and set the configuration
		MLLogisticRegression learner = new MLLogisticRegression(args[0]);

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
	public void savemodel(String fname) {
		// TODO Auto-generated method stub
		try{
			System.out.print( "Saving model (" + fname + ")..." );
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));
			    
			for(int i = 0; i< this.w.length; i++ ){
				writer.write( ""+ this.bias[i] );
				for(int j = 0; j< this.d; j++ ){
					writer.write( " "+ this.w[i][j] );
				}
				writer.write( "\n" );
			}
			
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
	public void loadmodel(String fname) {		
		try {
			System.out.println( "Loading model (" + fname + ")..." );
			Path p = Paths.get(fname);

			BufferedReader reader = Files.newBufferedReader(p, Charset.forName("UTF-8"));
		    String line = null;
		    
		    // read file
		    ArrayList<String> lines = new ArrayList<String>(); 
		    while ((line = reader.readLine()) != null) {
		        lines.add(line);
		    }
		    
		    reader.close();
		    
		    // process lines
		    // allocate the model
		    this.m = lines.size()-1;
		    this.w = new double[this.m][];
		    this.bias = new double[this.m];
		    
		    for( int i = 0; i < this.m; i++ ){
		    	String[] values =  lines.get(i).split( " " );
		    	this.w[i] = new double[values.length-1];
		    	
		    	this.bias[i] = Double.parseDouble(values[0]);
		    	for( int j=1; j < values.length; j++ ){
		    		this.w[i][j-1] = Double.parseDouble(values[j]);
		    	}
		    }
		    
		    this.d = this.w[0].length;
		    
		    // last line for thresholds
		    this.thresholds = new double[this.m];
		    String[] values =  lines.get(lines.size()-1).split( " " );
	    	for( int j=0; j < values.length; j++ ){
	    		this.thresholds[j] = Double.parseDouble(values[j]);
	    	}
		    
		    
		    
		    System.out.println( "Done." );
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
		
	}

}
