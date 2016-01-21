package threshold;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTEum extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTEum.class);

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
			ArrayList<ComparablePair> posteriors = new ArrayList<>();
			int[] labels = new int[data.n];
			
			for( int j = 0; j < data.n; j++ ) {
				double post = learner.getPosteriors(data.x[j], i);
				//logger.info ( post );
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
			
//			logger.info( "Class: " + i +" (" + numOfPositives + ")\t" 
//			                         +" F: " + String.format("%.4f", maxFmeasure ) 
//			                         + " Th: " + String.format("%.4f", maxthreshold) );
			
			thresholds[i] = maxthreshold;
			avgFmeasure += maxFmeasure;
			
		}
		logger.info( "Validated macro F-measure: {}", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		
		return thresholds;
	}

	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		// TODO Auto-generated method stub
		return null;
	}

	
}

