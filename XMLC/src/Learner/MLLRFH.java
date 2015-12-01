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
import preprocessing.FH;
import util.HashFunction;
import util.MasterSeed;

public class MLLRFH extends AbstractLearner {
	protected int epochs = 1;

	//protected DenseVector w = null;
	protected double[] w = null;
	//protected StepFunction[] stepfunctions;

	protected double gamma = 0; // learning rate
	protected int step = 0;
	protected double delta = 0;
    
	protected boolean geomWeighting = true;

	protected int T = 1;
	protected AVTable traindata = null;

	// 0 = "vanila"
	protected int updateMode = 0;

	Random shuffleRand;

	
	protected FH fh = null;
	
	protected int hd;

	protected double learningRate;
	
	//private double[] bias;
	
	public MLLRFH(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
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

		int seed = 1;
		this.hd = 2000000;
		
		
		this.fh = new FH(seed, this.hd, this.m);
		
		
		System.out.println( "Num. of labels: " + this.m + " Dim: " + this.d + " Hash dim: " + this.hd );
		Random allocationRand = MasterSeed.nextRandom();

		System.out.print( "Allocate the learners..." );

		//this.w = new DenseVector(this.hd);
		this.w = new double[this.hd];
		this.thresholds = new double[this.m];
		//this.bias = new double[this.m];
		//this.stepfunctions = new StepFunction[this.m];

		for (int i = 0; i < this.m; i++) {
			//this.w[i] = new DenseVector(this.hd + 1);
			//this.stepfunctions[i] = this.stepFunction.clone();
			this.thresholds[i] = 0.5;
		}
		
		//how to initialize w?
		
		System.out.println( "Done." );
	}
	
	
	protected double scalar = 1.0;
	protected double lambda = 0.001;

	protected void updatedPosteriors( int currIdx, int label, double inc, double y) {
	
		int n = traindata.x[currIdx].length;
		
		
		for(int i = 0; i < n; i++) {
			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			
			//double update = inc * traindata.x[currIdx][i].value;
			
			double gradient = inc * traindata.x[currIdx][i].value; //this.scalar * inc * traindata.x[currIdx][i].value;
						
			double update = this.learningRate * gradient; //(this.learningRate * gradient * (y*2.0 - 1) * traindata.x[currIdx][i].value) / this.scalar;		
					
			//System.out.println("w -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
			
			this.w[index] -= update; 
			//this.w.set(index, this.w.get(index) - update);
			
			
		}
		
		// Include bias term in weight vector:
		int biasIndex = fh.getIndex(label, -1);
		
		//double bias = this.w[biasIndex]; //this.w.get(fh.getIndex(label, -1));
		
		double gradient = inc; //this.scalar * inc;
		
		double update = this.learningRate * gradient;//(this.learningRate * gradient * (y*2.0 - 1))  / this.scalar;		
		
		//this.w.set(biasIndex, bias - update);
		
		this.w[biasIndex] -= update;
		
		//System.out.println("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);

		
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
		this.scalar = 1;
		
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			ArrayList<Integer> indirectIdx = this.shuffleIndex();

			for (int i = 0; i < traindata.n; i++) {
				
				
				this.learningRate = 1.0 / (Math.ceil(this.T / ((double) this.step)));
				//this.scalar *= (1 - this.learningRate * this.lambda);
				
				
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

					updatedPosteriors( currIdx, label, inc, currLabel);

				}

				this.T++;

				if ((i % 10000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					//System.out.println("Weight: " + this.w[0].get(0) );
				}

			}

			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );


			//save model !!!!!!!!!!!!!!!!!!!!!!!
			//String modelFile = this.getProperties().getProperty("ModelFile");
			//this.savemodel(modelFile);
		}
		
		int zeroW = 0;
		double sumW = 0;
		for(double weight : w) {
			if(weight == 0) zeroW++;
			sumW += weight;
		}
		
		System.out.println("Hash weights: " + w.length + ", " + zeroW + ", " + (w.length - zeroW) + ", " + sumW + ", ");
		

	}



	Sigmoid s = new Sigmoid();
	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		
		
		for (int i = 0; i < x.length; i++) {
		
			int hi = fh.getIndex(label,  x[i].index); 
			posterior += x[i].value * this.w[hi];//this.w.get(hi);
		}

		int hi = fh.getIndex(label,  -1); //this.taskhash[label].hash(-1); //-1 indicates w_0
		
		
		posterior += this.w[hi];//this.w.get(hi);
		
		
		posterior = s.value(posterior);
		
		//System.out.println("Posterior :" + posterior);
		
		return posterior;
	}

	@Override
	public void savemodel(String fname) {
		// TODO Auto-generated method stub
		try{
			System.out.print( "Saving model (" + fname + ")..." );
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));

			for(int i = 0; i< this.w.length; i++ ){
				writer.write( " "+ this.w[i]/*.get(i)*/ );
			}
			writer.write( "\n" );

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
		    this.w = new double[this.hd];//new DenseVector(this.hd);

		    
		    String[] values =  lines.get(0).split( " " );
		    double[] weights = new double[values.length];
		    for( int j=0; j < values.length; j++ ){
		    	weights[j] = Double.parseDouble(values[j]);
		    }
		    
		    this.d = weights.length;
		    this.w = weights;
		    
		    // last line for thresholds
		    this.thresholds = new double[this.m];
		    String[] tValues =  lines.get(lines.size()-1).split( " " );
	    	for( int j=0; j < values.length; j++ ){
	    		this.thresholds[j] = Double.parseDouble(tValues[j]);
	    	}



		    System.out.println( "Done." );
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}

	}

}
