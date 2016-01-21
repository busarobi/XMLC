package threshold;

import java.util.HashSet;
import java.util.Properties;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public abstract class ThresholdTuning {
	protected double[] thresholds = null;
	protected int m = 0;
	protected Properties properties = null;
	
	public ThresholdTuning( int m, Properties properties ) {
		this.properties = properties;
		
		this.m = m;
		thresholds = new double[m];		
	}
	
	public double getThreshold( int classIdx ) {
		return thresholds[classIdx];
	}
	
	abstract public double[] validate( AVTable data, AbstractLearner learner ); 
	abstract public double[] validate( AVTable data, AVTable sPEarray );
	
}
