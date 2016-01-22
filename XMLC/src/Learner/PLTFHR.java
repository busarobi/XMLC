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
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Learner.step.StepFunction;
import preprocessing.UniversalHasher;

public class PLTFHR extends PLTFH {
	private static final long serialVersionUID = 8959369680174721738L;

	private static Logger logger = LoggerFactory.getLogger(PLTFHR.class);

	protected int[] Tarray = null;	
	protected double[] scalararray = null;
	
	//protected int t = 0;
	//protected double innerThreshold = 0.15;

	//protected double[] scalars = null;
	
	public PLTFHR(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Leraner: PLTFTHR" );

		//this.innerThreshold = Double.parseDouble(this.properties.getProperty("IThreshold", "0.15") );
		//logger.info("#### Inner node threshold : " + this.innerThreshold );
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);

		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
		
		//logger.info( "Done." );
	}
	
		
	@Override
	public void train(AVTable data) {
		
		
				
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			
			for (int i = 0; i < traindata.n; i++) {
				int currIdx = indirectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				for (int j = 0; j < traindata.y[currIdx].length; j++) {

					int treeIndex = traindata.y[currIdx][j] + traindata.m - 1;
					positiveTreeIndices.add(treeIndex);

					while(treeIndex > 0) {

						treeIndex = (int) Math.floor((treeIndex - 1)/2);
						positiveTreeIndices.add(treeIndex);

					}
				}

				if(positiveTreeIndices.size() == 0) {

					negativeTreeIndices.add(0);

				} else {

					PriorityQueue<Integer> queue = new PriorityQueue<Integer>();
					queue.add(0);

					while(!queue.isEmpty()) {

						int node = queue.poll();
						int leftchild = 2 * node + 1;
						int rightchild = 2 * node + 2;

						Boolean left = false, right = false;

						if(positiveTreeIndices.contains(leftchild)) {
							queue.add(leftchild);
							left = true;
						}

						if(positiveTreeIndices.contains(rightchild)) {
							queue.add(rightchild);
							right = true;
						}

						if(left == true && right == false) {
							negativeTreeIndices.add(rightchild);
						}

						if(left == false && right == true) {
							negativeTreeIndices.add(leftchild);
						}

					}
				}

				//logger.info("Negative tree indices: " + negativeTreeIndices.toString());


				for(int j:positiveTreeIndices) {

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(1.0 - posterior); 

					updatedPosteriors(currIdx, j, inc);
				}

				for(int j:negativeTreeIndices) {

					if(j >= this.t) logger.info("ALARM");

					double posterior = getPartialPosteriors(traindata.x[currIdx],j);
					double inc = -(0.0 - posterior); 
					
					updatedPosteriors(currIdx, j, inc);
				}

		

				if ((i % 100000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					//logger.info("Weight: " + this.w[0].get(0) );
					logger.info("Scalar: " + this.scalararray[0]);
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
	public double getPartialPosteriors(AVPair[] x, int label) {
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
	
	
	public void save(String fname) {
		try{
			logger.info( "Saving model (" + fname + ")..." );						
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));

			writer.write( "d = "+ this.d + "\n" );
			writer.write( "hd = "+ this.hd + "\n" );
			writer.write( "m = "+ this.m + "\n" );
			
			// write out weights
			writer.write( ""+ this.w[0]/*.get(i)*/ );
			for(int i = 1; i< this.w.length; i++ ){
				writer.write( " "+ this.w[i]/*.get(i)*/ );
			}
			writer.write( "\n" );

			// bias
			writer.write( ""+ this.bias[0]/*.get(i)*/ );
			for(int i = 1; i< this.bias.length; i++ ){
				writer.write( " "+ this.bias[i]/*.get(i)*/ );
			}
			writer.write( "\n" );
						
			// write out threshold
			writer.write( ""+ this.thresholds[0] );
			for(int i = 1; i< this.thresholds.length; i++ ){
				writer.write( " "+ this.thresholds[i] );
			}
			writer.write( "\n" );

			
			// scalar
			writer.write( ""+ this.scalararray[0]/*.get(i)*/ );
			for(int i = 1; i< this.scalararray.length; i++ ){
				writer.write( " "+ this.scalararray[i]/*.get(i)*/ );
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
		    values =  lines.get(5).split( " " );
		    this.thresholds = new double[values.length];
	    	for( int j=0; j < values.length; j++ ){
	    		this.thresholds[j] = Double.parseDouble(values[j]);
	    	}


	    	// scalar values
		    values =  lines.get(6).split( " " );
		    this.scalararray = new double[values.length];
	    	for( int j=0; j < values.length; j++ ){
	    		this.scalararray[j] = Double.parseDouble(values[j]);
	    	}
	    	
	    	
	    	this.t = 2 * this.m - 1;
			this.fh = new UniversalHasher(this.fhseed, this.hd, this.t);


	    	this.scalar=1.0;
		    logger.info( "Done." );
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
		
	}
	
	
	
}
