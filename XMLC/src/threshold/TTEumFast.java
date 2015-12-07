package threshold;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Properties;

import Data.AVTable;
import Data.ComparablePair;
import Data.ComparableTriplet;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTEumFast extends ThresholdTuning {

	public TTEumFast(int m, Properties properties) {
		super(m, properties );		
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		
		double minThreshold = 0.001;
		
		thresholds = new double[this.m];
		
		double avgFmeasure = 0.0;

		// for labels
		//int[] indices = new int[data.n];
		//for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		ArrayList<ComparableTriplet>[] posteriors = new ArrayList[learner.getNumberOfLabels()];
		int [] numPositives = new int[learner.getNumberOfLabels()];
		
		for( int j = 0; j < data.n; j++ ) {
			
			HashSet<Integer> trueLabels = new HashSet<Integer>();
			
			for(int m = 0; m < data.y[j].length; m++) {
				trueLabels.add(data.y[j][m]);
			}
			
			HashSet<EstimatePair> sPE = learner.getSparseProbabilityEstimates(data.x[j], minThreshold);
			
			for(EstimatePair pred : sPE) {
				
				int y = 0;
				int label = pred.getLabel();
				if(trueLabels.contains(label)) {
					y=1;
					numPositives[label]++;
				}
				
				if(posteriors[pred.getLabel()] == null) {
					posteriors[pred.getLabel()] = new ArrayList<ComparableTriplet>();
				}
				posteriors[pred.getLabel()].add(new ComparableTriplet(pred.getP(), j, y));				
			}
		}
	
		for(int i = 0; i < posteriors.length; i++) {
		
			if(posteriors[i] == null) {
				thresholds[i] = 0.5;
				continue;
			}
			
			Collections.sort(posteriors[i]);

			// tune the threshold			
			// every instance is predicted as positive first with a threshold = 0.5
			double maxthreshold = 1.0;
			int tp = 0;
			int predictedPositives = 0;
			double Fmeasure = 0; //((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives )); 
			double maxFmeasure = 0; //Fmeasure;

			
			//predictedPositives = posteriors[i].size();
			
			for(ComparableTriplet triplet: posteriors[i]) {
				
				//System.out.println(triplet.getKey());
				
				if(triplet.gety() == 1) {
					tp++;
					
				}
				
				predictedPositives++;
				
				Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = triplet.getKey();
				}
			}
			
			
			tp = numPositives[i];
			predictedPositives = data.n;
			Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives )); 
			
			if (maxFmeasure < Fmeasure ) {
				maxthreshold = 0.0;
				maxFmeasure = Fmeasure;
			}
		
				
			//System.out.println(maxthreshold + " " + maxFmeasure);
			thresholds[i] = Math.min(0.5, maxthreshold);
			avgFmeasure += maxFmeasure;
		
			System.out.println("Label: " + i + " threshold: " + thresholds[i] + " F: " + maxFmeasure);
		}

		System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		
		
		
		return thresholds;
	}
	
	
}

