package threshold;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Properties;

import Data.AVTable;
import Data.ComparablePair;
import Learner.AbstractLearner;

public class TTEum extends ThresholdTuning {
		
	public TTEum(int m, Properties properties) {
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
			ArrayList<ComparablePairEUM> posteriors = new ArrayList<>();
			int[] labels = new int[data.n];
			
			for( int j = 0; j < data.n; j++ ) {
				double post = learner.getPosteriors(data.x[j], i);
				//System.out.println ( post );
				ComparablePairEUM entry = new ComparablePairEUM( post, j);
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
			double Fmeasure = ((2.0*tp)) / ((double) ( numOfPositives + predictedPositives )); 
			double maxthreshold = 0.0;
			double maxFmeasure = Fmeasure;

			
			for( int j = 0; j < data.n; j++ ) {				
				int ind = posteriors.get(j).getValue();
				if ( labels[ind] == 1 ){
					tp--;
				}
				predictedPositives--;
				
				Fmeasure = ((2.0*tp)) / ((double) ( numOfPositives + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = posteriors.get(j).getKey();
				}
			}			
			
//			System.out.println( "Class: " + i +" (" + numOfPositives + ")\t" 
//			                         +" F: " + String.format("%.4f", maxFmeasure ) 
//			                         + " Th: " + String.format("%.4f", maxthreshold) );
			
			thresholds[i] = maxthreshold;
			avgFmeasure += maxFmeasure;
			
		}
		System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		
		return thresholds;
	}
	
	
}

