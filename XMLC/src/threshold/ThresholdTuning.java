package threshold;

import Data.AVTable;
import Learner.AbstractLearner;

public abstract class ThresholdTuning {
	protected double[] thresholds = null;
	protected int m = 0;
	
	public ThresholdTuning( int m ) {
		this.m = m;
		thresholds = new double[m];
	}
	
	public double getThreshold( int classIdx ) {
		return thresholds[classIdx];
	}
	
	abstract public double[] validate( AVTable data, AbstractLearner learner ); 
	
	
}
