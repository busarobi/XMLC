package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
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

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import IO.DataReader;
import IO.Evaluator;
import IO.Result;
import preprocessing.FeatureHasher;

public class MLLogisitcRegression extends AbstractLearner {
	protected int epochs = 20;

	protected double[][] w = null;
	protected double[][] grad = null;
	protected double[] bias = null;
	protected double[] gradbias = null;

	protected double gamma = 0; // learning rate
	protected int step = 0;
	protected double delta = 0;
    protected int OFOepochs = 1;
	// uniform sampling of negatives, the number of negatives is r times more as
	// many as positives
	protected int r = 1;

	protected int T = 1;	
	protected AVTable traindata = null;

	// 0 = "vanila"
	protected int updateMode = 0;

	protected Random rand = new Random();

	public MLLogisitcRegression(String propertyFileName) {
		super(propertyFileName);
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

		System.out.println( "Num. of labels: " + this.m + " Dim: " + this.d );
		
		this.w = new double[this.m][];
		this.bias = new double[this.m];

		this.grad = new double[this.m][];
		this.gradbias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.w[i] = new double[d];
			this.grad[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * rand.nextDouble() - 1.0;

			this.bias[i] = 2.0 * rand.nextDouble() - 1.0;

			if ((i % 100) == 0)
				System.out.println( "Model: "+ i +" (" + this.m + ")" );

		}

		this.thresholds = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.thresholds[i] = 0.2;
		}
		
		// learning rate
		this.gamma = 10.0;
		// step size for learning rate
		this.step = 2000;
		this.T = 1;
		// decay of gradient
		this.delta = 0.0;
				
		this.epochs = 30;
	}

	@Override
	public void train(AVTable data) {
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			
			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>();
			for (int i = 0; i < this.traindata.n; i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx);

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
			String modelFile = this.getProperties().getProperty("ModelFile");
			this.savemodel(modelFile);
		}

	}

	
	public void validateThresholdEUM( AVTable data ) {
		
		double avgFmeasure = 0.0;
		// for labels
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		for( int i = 0; i < this.m; i++ ) {
			ArrayList<ComparablePair> posteriors = new ArrayList<>();
			int[] labels = new int[data.n];
			
			for( int j = 0; j < data.n; j++ ) {
				double post = this.getPosteriors(data.x[j], i);
				//System.out.println ( post );
				ComparablePair entry = new ComparablePair( post, j);
				posteriors.add(entry);
			}
			
			Collections.sort(posteriors);
			
			// assume that the labels are ordered
			int numOfPositives = 0;
			for( int j = 0; j < data.n; j++ ) {
				if ( (indices[j] < data.y[j].length) &&  (data.y[j][indices[j]] == i) ){ 
					labels[j] = 1;
				    numOfPositives++;
				    indices[j]++;
				} else {
					labels[j] = 0;
//					if ((indices[j] < data.y[j].length) && (data.y[j][indices[j]] < j ) ) 
//						indices[j]++;
				}
			}

			// tune the threshold			
			// every instance is predicted as positive first with a threshold = 0.0
			int tp = numOfPositives;
			int predictedPositives = data.n;
			double Fmeasure = ((double) tp) / ((double) ( numOfPositives + predictedPositives )); 
			double maxthreshold = 0.0;
			double maxFmeasure = Fmeasure;

			
			for( int j = 0; j < data.n; j++ ) {				
				int ind = posteriors.get(j).getValue();
				if ( labels[ind] == 1 ){
					tp--;
				}
				predictedPositives--;
				
				Fmeasure = ((double) tp) / ((double) ( numOfPositives + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = posteriors.get(j).getKey();
				}
			}			
			
			System.out.println( "Class: " + i +" (" + numOfPositives + ")\t" 
			                         +" F: " + String.format("%.4f", maxFmeasure ) 
			                         + " Th: " + String.format("%.4f", maxthreshold) );
			
			this.thresholds[i] = maxthreshold;
			avgFmeasure += maxFmeasure;
			
		}
		
		System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) this.m) ) ;
		
	}
	

	
	public void validateThresholdOFO( AVTable data ) {		
		int[] TP = new int[this.m];
		int[] P = new int[this.m];
		int[] PredP = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		//double[] prior = AVTable.getPrior(data);
		
		for( int i = 0; i < this.m; i++ ) {
			TP[i] = 0;
			P[i] = 0;
			PredP[i] = 0;
						
			this.thresholds[i] = ((double) TP[i]) / ((double) P[i] + PredP[i]);
		}
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
			for (int i = 0; i < this.m; i++) {

				// assume that the labels are ordered
				int currentLabel = 0;
				for (int j = 0; j < data.n; j++) {
					if ((indices[j] < data.y[j].length) && (data.y[j][indices[j]] == i)) {
						currentLabel = 1;
						indices[j]++;
					} else {
						currentLabel = 0;
					}

					double post = this.getPosteriors(data.x[j], i);
					if (post > this.thresholds[i]) {
						PredP[i]++;
						if (currentLabel == 1)
							TP[i]++;
					}
					if (currentLabel == 1)
						P[i]++;

					this.thresholds[i] = ((double) TP[i]) / ((double) P[i] + PredP[i]);
				}
												
			}

		}
		
		for( int i=0; i < this.m; i++ )
			System.out.println( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Old OFO
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	public void validateThresholdEXU( AVTable data ) {		
		int[] at = new int[this.m];
		int[] bt = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		double[] prior = AVTable.getPrior(data);
		
		for( int i = 0; i < this.m; i++ ) {
			//at[i] = (int) Math.round(prior[i] * 1000);
			//bt[i] = 1000;
			at[i] = 0;
			bt[i] = 0;
			
			double F00 = (2.0 * at[i]) / ((double) bt[i]);
			double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
			double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);
			
			
			this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
		}
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
			for (int i = 0; i < this.m; i++) {

				// assume that the labels are ordered
				int currentLabel = 0;
				for (int j = 0; j < data.n; j++) {
					if ((indices[j] < data.y[j].length) && (data.y[j][indices[j]] == i)) {
						currentLabel = 1;
						indices[j]++;
					} else {
						currentLabel = 0;
					}

					double post = this.getPosteriors(data.x[j], i);
					if (post > this.thresholds[i]) {
						bt[i]++;
						if (currentLabel == 1)
							at[i]++;
					}
					if (currentLabel == 1)
						bt[i]++;

					double F00 = (2.0 * at[i]) / ((double) bt[i]);
					double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
					double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);
					
					
					this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
					//this.thresholds[i] = ((prior[i] * (F01 - F00)) + F00) / ((prior[i] * (2*F01 - F11 ) ) + F00 + F01 );
				}
												
			}

		}
		
		for( int i=0; i < this.m; i++ )
			System.out.println( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		
	}
	
	
	
	
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 0.0;
		for (int i = 0; i < x.length; i++) {
			posterior += (x[i].value * this.w[label][x[i].index]);
		}
		posterior += this.bias[label];
		posterior = 1.0 / (1.0 + Math.exp(-posterior));
		return posterior;
	}

	@Override
	public Evaluator test(AVTable data) {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// create the classifier and set the configuration
		MLLogisitcRegression learner = new MLLogisitcRegression(args[0]);

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
		
		
		
		
		// evaluate (EUM)
		 
		learner.validateThresholdEUM(data);
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);
		for ( String perfName : perf.keySet() ) {
			System.out.println("##### " + perfName + ": "  + perf.get(perfName));
		}
        
		
		// evaluate (OFO)
//		learner.validateThresholdOFO(data);
//		Map<String,Double> perfOFO = Evaluator.computePerformanceMetrics(learner, testdata);
//		for ( String perfName : perfOFO.keySet() ) {
//			System.out.println("##### " + perfName + ": "  + perfOFO.get(perfName));
//		}

		// evaluate (EXU)
//		learner.validateThresholdEXU(data);
//		Map<String,Double> perfEXU = Evaluator.computePerformanceMetrics(learner, testdata);
//		for ( String perfName : perfEXU.keySet() ) {
//			System.out.println("##### " + perfName + ": "  + perfEXU.get(perfName));
//		}
		
		
	}

	@Override
	public void savemodel(String fname) {
		// TODO Auto-generated method stub
		try{
			System.out.print( "Saving model (" + fname + ")..." );
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));
			    
			for(int i = 0; i< this.m; i++ ){
				writer.write( ""+ this.bias[i] );
				for(int j = 0; j< this.d; j++ ){
					writer.write( " "+ this.w[i][j] );
				}
				writer.write( "\n" );
			}
			
			writer.write( ""+ this.thresholds[0] );
			for(int i = 1; i< this.m; i++ ){
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

			BufferedReader reader = Files.newBufferedReader(p);
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
