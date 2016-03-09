package Learner;


import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Learner.step.StepFunction;
import Learner.step.StepFunctionFactory;
import threshold.ThresholdTuning;
import util.IoUtils;


public abstract class AbstractLearner implements Serializable{
	private static final long serialVersionUID = -1399552145906714507L;

	private static Logger logger = LoggerFactory.getLogger(AbstractLearner.class);

	protected int m = 0; // num of labels
	protected int d = 0; // number of features


	transient protected Properties properties = null;
	protected double[] thresholds = null;
	transient protected StepFunction stepFunction;
	// abstract functions
	public abstract void allocateClassifiers( AVTable data );
	public abstract void train( AVTable data );
	//public abstract Evaluator test( AVTable data );
	public abstract double getPosteriors(AVPair[] x, int label);

	public void savemodel(String fname ) throws IOException{
		IoUtils.serialize(this, Paths.get(fname));
	}
	public static AbstractLearner loadmodel(String fname ) throws FileNotFoundException, ClassNotFoundException, IOException{
		return (AbstractLearner) IoUtils.deserialize(Paths.get(fname));
	}

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
		logger.info("--> Learner: {}", learnerName);
		if (learnerName.compareTo("MLLog")==0)
			learner = new MLLogisticRegression(properties, stepfunction);
		else if (learnerName.compareTo( "Constant" ) == 0)
			learner = new ConstantLearner(properties, stepfunction);
		else if (learnerName.compareTo("MLLogNP") == 0)
			learner = new MLLogisticRegressionNSampling(properties, stepfunction);
		else if (learnerName.compareTo("MLLRFH") == 0)
			learner = new MLLRFH(properties, stepfunction);
		else if (learnerName.compareTo("MLLRFHNS") == 0)
			learner = new MLLRFHNS(properties, stepfunction);
		else if (learnerName.compareTo("MLLRFHR") == 0)
			learner = new MLLRFHR(properties, stepfunction);
		else if (learnerName.compareTo("PLTFHKary") == 0)
			learner = new PLTFHKary(properties, stepfunction);
		else if (learnerName.compareTo("PLTFHRKary") == 0)
			learner = new PLTFHRKary(properties, stepfunction);
		else if (learnerName.compareTo("PLTFH") == 0)
			learner = new PLTFH(properties, stepfunction);		
		else if (learnerName.compareTo("PLTFHR") == 0)
			learner = new PLTFHR(properties, stepfunction);		
		else if (learnerName.compareTo("PLT") == 0)
			learner = new PLT(properties, stepfunction);
		else if (learnerName.compareTo("BRTFHR") == 0)
			learner = new BRTFHR(properties, stepfunction);
		else if (learnerName.compareTo("BRTFHRNS") == 0)
			learner = new BRTFHRNS(properties, stepfunction);
		else if (learnerName.compareTo("BRTreeFHRNS") == 0)
			learner = new BRTreeFHRNS(properties, stepfunction);
		else {
			System.err.println("Unknown learner");
			System.exit(-1);
		}
				
		return learner;		
				
	}
	
	//public void tuneThreshold( ThresholdTuning t, AVTable data ){
	//	this.thresholds = t.validate(data, this);
	//}
	
	public void tuneThreshold( ThresholdTuning t, AVTable data ){
		this.setThresholds(t.validate(data, this));
	}

	public void setThresholds(double[] t) {		
		for(int j = 0; j < t.length; j++) {
			this.thresholds[j] = t[j];
		}		
	}

	public void setThresholds(double t) {		
		for(int j = 0; j < this.thresholds.length; j++) {
			this.thresholds[j] = t;
		}		
	}
	
	public void setThreshold(int label, double t) {
		this.thresholds[label] = t;
	}	
	
	public AbstractLearner(Properties properties, StepFunction stepfunction){
		this.properties = properties;
		this.stepFunction = stepfunction;
	}


	// naive implementation checking all labels
	public HashSet<Integer> getPositiveLabels(AVPair[] x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		for( int i = 0; i < this.m; i++ ) {
			if (this.getPosteriors(x, i) >= this.thresholds[i]) {
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
	
	public int[] getTopkLabels(AVPair[] x, int k) {
		PriorityQueue<ComparablePair> pq = new PriorityQueue<ComparablePair>();
		
		for( int i = 0; i < this.m; i++ ) {
			double post = this.getPosteriors(x, i);
			pq.add(new ComparablePair(post, i));
		}
		
		
		int[] labels = new int[k];
		for( int i=0; i<k; i++ ){
			ComparablePair p = pq.poll();
			labels[i] = p.getValue();
		}
		
		return labels;
	}
	
	
	public void outputPosteriors( String fname, AVTable data )
	{
		try{
			logger.info( "Saving posteriors (" + fname + ")..." );
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
			logger.info( "Done." );
		} catch (IOException e) {
			logger.info(e.getMessage());
		}
		
	}
	
	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold) {
		
		HashSet<EstimatePair> positiveLabels = new HashSet<EstimatePair>();
		
		for(int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);
			if (p >= threshold) 
				positiveLabels.add(new EstimatePair(i, p));
		}
		
		return positiveLabels;
	}
	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();
		
		for(int i = 0; i < this.m; i++) {
			double p = getPosteriors(x, i);			 
			positiveLabels.add(new EstimatePair(i, p));
		}
		
		while( positiveLabels.size()>=k){			
			positiveLabels.pollLast();
		}
					
		return positiveLabels;
		
	}
		
	public Properties getProperties() {
		return properties;
	}

	public int getNumberOfLabels() {
		return this.m;
	}
	
}
