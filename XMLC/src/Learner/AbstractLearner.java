package Learner;


import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Learner.step.StepFunction;
import Learner.step.StepFunctionFactory;
import threshold.ThresholdTuning;


public abstract class AbstractLearner {
	protected int m = 0; // num of labels
	protected int d = 0; // number of features


	protected Properties properties = null;
	protected double[] thresholds = null;
	protected StepFunction stepFunction;
	// abstract functions
	public abstract void allocateClassifiers( AVTable data );
	public abstract void train( AVTable data );
	//public abstract Evaluator test( AVTable data );
	public abstract double getPosteriors(AVPair[] x, int label);

	public abstract void savemodel(String fname );
	public abstract void loadmodel(String fname );

	public int getPrediction(AVPair[] x, int label){
		if ( this.thresholds[label] <= getPosteriors(x, label) ) {
			return 1;
		} else {
			return 0;
		}
	}

	
	public static AbstractLearner learnerFactory( Properties properties ) {
		AbstractLearner learner = null;
		
		StepFunction stepfunction = StepFunctionFactory.factory(properties);
		
		String learnerName = properties.getProperty("Learner");
		System.out.println("--> Learner: " + learnerName);
		if (learnerName.compareTo("MLLog")==0)
			learner = new MLLogisticRegression(properties, stepfunction);
		else if (learnerName.compareTo( "Constant" ) == 0)
			learner = new ConstantLearner(properties, stepfunction);
		else if (learnerName.compareTo("MLLogNP") == 0)
			learner = new MLLogisticRegressionNSampling(properties, stepfunction);
		else if (learnerName.compareTo("MLLRFH") == 0)
			learner = new MLLRFH(properties, stepfunction);
		else if (learnerName.compareTo("PLTFH") == 0)
			learner = new PLTFH(properties, stepfunction);		
		else if (learnerName.compareTo("PLT") == 0)
			learner = new PLT(properties, stepfunction);
		else {
			System.err.println("Unknown learner");
			System.exit(-1);
		}
				
		return learner;		
				
	}
	
	
	public void tuneThreshold( ThresholdTuning t, AVTable data ){
		this.thresholds = t.validate(data, this);
	}


	public AbstractLearner(Properties properties, StepFunction stepfunction){
		this.properties = properties;
		this.stepFunction = stepfunction;
	}


	// naive implementation checking all labels
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		for( int i = 0; i < this.m; i++ ) {
			if ( 0 < this.getPrediction(x, i) ){
				positiveLabels.add(i);
			}
		}

		return positiveLabels;
	}
	
	// naive implementation checking all labels
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();

		for( int i = 0; i < this.m; i++ ) {
			double post = getPosteriors(x, i);
			if ( this.thresholds[i] <= post ){
				positiveLabels.add(new ComparablePair(post, i ));
			}
		}

		return positiveLabels;
	}
	

	
	
	public void outputPosteriors( String fname, AVTable data )
	{
		try{
			System.out.print( "Saving posteriors (" + fname + ")..." );
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));
			    
			for(int i = 0; i< data.n; i++ ){
				//HashSet<Integer> predictedLabels = this.getPositiveLabels(data.x[i]);
				//List<Integer> sortedList = new ArrayList<Integer>(predictedLabels);
				//Collections.sort(sortedList);
				//for( int j : sortedList ){
				for( int j = 0; j < this.m; j++ ){
					writer.write( j + ":" + this.getPosteriors(data.x[i], j) + " " );
				}
				writer.write( "\n" );
			}
						
			writer.close();
			System.out.println( "Done." );
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
		
	}
	
	
		
	public Properties getProperties() {
		return properties;
	}

	public int getNumberOfLabels() {
		return this.m;
	}
}
