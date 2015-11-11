package Learner;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Properties;

import Data.AVPair;
import Data.AVTable;
import IO.Evaluator;
import threshold.ThresholdTuning;


public abstract class AbstractLearner {
	protected int m = 0; // num of labels
	protected int d = 0; // number of features
	
	
	protected Properties properties = null;
	protected double[] thresholds = null;
	// abstract functions
	public abstract void allocateClassifiers( AVTable data );
	public abstract void train( AVTable data );
	//public abstract Evaluator test( AVTable data );
	public abstract double getPosteriors(AVPair[] x, int label);
	
	public abstract void savemodel(String fname );
	public abstract void loadmodel(String fname );
	
	public int getPrediction(AVPair[] x, int label){
		if ( this.thresholds[label] < getPosteriors(x, label) )
			return 1;
		else return 0;
	}

	public void tuneThreshold( ThresholdTuning t, AVTable data ){
		this.thresholds = t.validate(data, this);
	}
	
	
	public AbstractLearner( String propertyFile ){
		this.readProperty( propertyFile );
	}
	
	
	// naive implemntation checking all labels
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();
		
		for( int i = 0; i < this.m; i++ ) {
			if ( 0 < this.getPrediction(x, i) ){
				positiveLabels.add(i);
			}
		}
		
		return positiveLabels;
	}
	
	
	
	public void readProperty(String fname) {
		System.out.print("Reading property file...");
		properties = new Properties();
		try {
			FileInputStream in = new FileInputStream(fname);
			properties.load(in);
			in.close();
		} catch (FileNotFoundException e) {
			System.err.println(e.getMessage());
			System.exit(-1);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			System.exit(-1);
		}
		System.out.println("Done.");

	}
	
	public Properties getProperties() {
		return properties;
	}
	
	public int getNumberOfLabels() {
		return this.m;
	}
}
