package Learner;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;

import Data.AVPair;
import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import IO.DataManager;
import IO.DataReader;


public class ConstantLearner extends AbstractLearner {

	private static final long serialVersionUID = 579975688059937300L;

	public ConstantLearner(Properties properties) {
		super(properties);
		// TODO Auto-generated constructor stub
	}

	public int getPrediction(AVPair[] x, int label){
		return 0;
	}
	
	public HashSet<EstimatePair> getSparseProbabilityEstimates(AVPair[] x, double threshold){
		return new HashSet<EstimatePair>();
	}
	
	
	// naive implementation checking all labels
	public PriorityQueue<ComparablePair> getPositiveLabelsAndPosteriors(AVPair[] x) {
		PriorityQueue<ComparablePair> positiveLabels = new PriorityQueue<>();
		return positiveLabels;
	}

	
	@Override
	public void allocateClassifiers(DataManager data) {
		// TODO Auto-generated method stub		
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfLabels();
	}

	@Override
	public void train(DataManager data) {
		// TODO Auto-generated method stub

	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		// TODO Auto-generated method stub
		return 0;
	}
}
