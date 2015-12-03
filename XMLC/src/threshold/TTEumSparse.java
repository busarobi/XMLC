package threshold;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;

import Data.AVTable;
import Data.ComparablePair;
import Learner.AbstractLearner;

public class TTEumSparse extends ThresholdTuning {

	public TTEumSparse(int m, Properties properties) {
		super(m, properties );		
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		thresholds = new double[this.m];
		double avgFmeasure = 0.0;

		// for labels
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		for( int i = 0; i < learner.getNumberOfLabels(); i++ ) {
		}			
			
		
		return thresholds;
		
	}
}
