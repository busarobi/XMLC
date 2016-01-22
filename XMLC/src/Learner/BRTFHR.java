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
import java.util.LinkedList;
import java.util.Properties;
import java.util.Queue;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Learner.step.StepFunction;
import preprocessing.MurmurHasher;
import preprocessing.UniversalHasher;

public class BRTFHR extends MLLRFHR {
	private static final long serialVersionUID = 6513552130781797264L;

	private static Logger logger = LoggerFactory.getLogger(BRTFHR.class);

	protected int[] Tarray = null;	
	protected double[] scalararray = null;
	protected int t = 0;
	transient Sigmoid s = new Sigmoid();

	
	public BRTFHR(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Leraner: BRTFHR" );
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {

		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		this.t = 2 * this.m - 1;

		logger.info( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info( "#### Num. of inner node of the trees: " + this.t  );
		logger.info("#####################################################" );
			
		if ( this.hasher.compareTo("Universal") == 0 ) {			
			this.fh = new UniversalHasher(fhseed, this.hd, this.t);
		} else if ( this.hasher.compareTo("Murmur") == 0 ) {
			this.fh = new MurmurHasher(fhseed, this.hd, this.t);
		} else {
			logger.info("Unknown hasher");
			System.exit(-1);
		}
		
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}

		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
		
		logger.info( "Done." );
	}
	
		
	@Override
	public void train(AVTable data) {
				
		//int[] v = new int[this.m - 1];
		int[] v = new int[this.t];
		
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			if(ep == 9000) {
				logger.info("Stop");
			}
			
			for (int i = 0; i < traindata.n; i++) {
				int currIdx = indirectIdx.get(i);
				
				Arrays.fill(v, 0);//int[] v = new int[this.t];
				
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					int treeIndex = traindata.y[currIdx][j] + this.m - 1;
					v[treeIndex] = 1;
				}
				
				for(int j = this.t - 1; j >= 0; j--) {
					
					double posterior = getPartialPosteriors(traindata.x[currIdx], j);
					
					double inc = posterior -  ((v[j] > 0) ? 1.0 : 0.0); 

					updatedPosteriors(currIdx, j, inc, v[j] > 1 ? v[j] : 1);
				
					if(j > 0) {
						int treeIndex = (j - 1) >> 1; 
						v[treeIndex] += v[j]; //Math.max(v[treeIndex], v[j]);// += v[j];
					}
				}
				
				
				/*
				
				int indexy = 0;
				
				for (int label = 0; label < this.m; label++) {										
					
					int treeIndex = label + this.m - 1;
					
					double posterior = getPartialPosteriors(traindata.x[currIdx], treeIndex);

					double currLabel = 0.0;
					
					if ((indexy < traindata.y[currIdx].length) && (traindata.y[currIdx][indexy] == (label))) {
						currLabel = 1.0;
						indexy++;
					}
					
					if(posterior >= thresholds[label]) {
						//v[(treeIndex - 1) >> 1]++;
						v[(int) Math.floor((treeIndex - 1)/2)]++;
					}

					// update the models
					double inc = posterior - currLabel;
					
					if (ep < 100 ) updatedPosteriors( currIdx, treeIndex, inc, 1.0);

				}
				
				if(ep >= 0) { 
				
					for(int j = this.m - 2; j >= 0; j--) {
					
						double posterior = getPartialPosteriors(traindata.x[currIdx],j);
						double inc = posterior -  ((v[j] > 0) ? 1.0 : 0.0); 

						updatedPosteriors(currIdx, j, inc, v[j] > 1 ? v[j] : 1);
					
						if((j > 0) && (posterior >= thresholds[j]) && (v[j] > 0)) {
							//v[(j - 1) >> 1] += v[j];
							v[(int) Math.floor((j - 1)/2)] += v[j];
						}
					}
				
				}
				
				*/
				
				if ((i % 10000) == 0) {
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


	protected void updatedPosteriors( int currIdx, int label, double inc, double weights) {
	
		
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);		
			this.w[index] -= weights * update; 
			
		}
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= weights * update;

	}

	
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
	
	@Override
	public double getPosteriors(AVPair[] x, int label) {

		int treeIndex = label + this.m - 1;
		return  getPartialPosteriors(x, treeIndex);
	
	}
	
	
	@Override
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {

		HashSet<Integer> positiveLabels = new HashSet<Integer>();
	    
		Queue<Integer> queue = new LinkedList<Integer>();

		queue.add(0);

		while(!queue.isEmpty()) {

			int node = queue.poll();

			double currentP = getPartialPosteriors(x, node);

			if(currentP >= this.thresholds[node]) {

				if(node < this.m - 1) {
					int leftchild = (node << 1) + 1;
					int rightchild = (node << 1) + 2;

					queue.add(leftchild);
					queue.add(rightchild);

				} else {

					positiveLabels.add(node - this.m + 1);

				}
			}
		}

		//logger.info("Predicted labels: " + positiveLabels.toString());

		return positiveLabels;
	}

	@Override
	public void setThreshold(int label, double t) {
		
		int treeIndex = label + this.m - 1;
		this.thresholds[treeIndex] = t;
		
		while(treeIndex > 0) {

			treeIndex =  (treeIndex-1) >> 1; //(int) Math.floor((treeIndex - 1)/2); //
			this.thresholds[treeIndex] = Math.min(this.thresholds[(treeIndex<<1)+1], this.thresholds[(treeIndex<<1)+2]);
		}
		
	}

	public void setThresholds(double t) {		
		for(int j = this.m - 2; j >= 0; j--) {
			this.thresholds[j] = t;
		}		
		
		//logger.info(Arrays.toString(this.thresholds));
	}

	
	
	public void setThresholds(double[] t) {
		
		for(int j = 0; j < t.length; j++) {
			this.thresholds[j + this.m - 1] = t[j];
		}
		
		for(int j = this.m - 2; j >= 0; j--) {
			this.thresholds[j] = Math.min(this.thresholds[2*j+1], this.thresholds[2*j+2]);
		}
		
		//for( int i=0; i < this.thresholds.length; i++ )
		//	logger.info( "Threshold: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		
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
