package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
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
		System.out.println("Tuning threshold (TTeumFast)...");
	
		System.out.println( "\t --> EUM fast starts" );
		DateFormat dateFormat1 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date1 = new Date();
		System.out.println("\t\t" + dateFormat1.format(date1));
		
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

			if ((j % 100000) == 0) {
				System.out.println( "\t --> Instance: " + j +" (" + data.n + ")" );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				System.out.println("\t\t" + dateFormat.format(date));
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
				maxthreshold = minThreshold; //0.0;
				maxFmeasure = Fmeasure;
			}
				
			//System.out.println(maxthreshold + " " + maxFmeasure);
			thresholds[i] = Math.min(1.0, maxthreshold);
			avgFmeasure += maxFmeasure;
		
			//System.out.println("Label: " + i + " threshold: " + thresholds[i] + " F: " + maxFmeasure);
		}

//		for( int i=0; i < this.m; i++ )
//			System.out.println( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		
		System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		

		
		System.out.println( "\t --> EUM fast end" );
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		System.out.println("\t\t" + dateFormat.format(date));
		
		
		return thresholds;
	}
	
	
}

