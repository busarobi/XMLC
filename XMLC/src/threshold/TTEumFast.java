package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparableTriplet;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTEumFast extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTEumFast.class);

	
	protected double minThreshold = 0.001;
	protected double[] minThresholdArray = null;
	
	public TTEumFast(int m, Properties properties) {
		super(m, properties );	
		
		this.minThreshold = Double.parseDouble(properties.getProperty("minThreshold", "0.001") );
		
		logger.info("#####################################################" );
		logger.info("#### EUM fast" );
		logger.info("#### Min threshold: " + this.minThreshold );
		logger.info("#####################################################" );		
		
//		this.minThresholdArray = new double[this.m];
//		for( int i = 0; i < this.m; i++ )
//			this.minThresholdArray[i] = this.minThreshold;
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		logger.info("Tuning threshold (TTeumFast)...");
	
		logger.info( "\t --> @@@@@@@@@@@@@@@@@@@ EUM fast starts" );
		DateFormat dateFormat1 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date1 = new Date();
		logger.info("\t\t" + dateFormat1.format(date1));
		
		thresholds = new double[this.m];
		
		double avgFmeasure = 0.0;

		// for labels
		//int[] indices = new int[data.n];
		//for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		ArrayList<ComparableTriplet>[] posteriors = new ArrayList[learner.getNumberOfLabels()];
		int [] numPositives = new int[learner.getNumberOfLabels()];
		
		int numOfPositives = 0;
		
		for( int j = 0; j < data.n; j++ ) {
			
			HashSet<Integer> trueLabels = new HashSet<Integer>();
			
			for(int m = 0; m < data.y[j].length; m++) {
				trueLabels.add(data.y[j][m]);
			}
			HashSet<EstimatePair> sPE = null;
			if (this.minThresholdArray==null) {
				sPE = learner.getSparseProbabilityEstimates(data.x[j], minThreshold);
			} else {
				learner.setThresholds(this.minThresholdArray);
				//sPE = learner.getSparseProbabilityEstimates(data.x[j]);
			}
			
			numOfPositives += sPE.size();
			
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
				logger.info( "\t --> Instance: " + j +" (" + data.n + ")" );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (j+1) );
			}
			
		}
	
		for(int i = 0; i < posteriors.length; i++) {
		
			if(posteriors[i] == null) {
				thresholds[i] = this.minThreshold;
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
				
				//logger.info(triplet.getKey());
				
				if(triplet.gety() == 1) {
					tp++;
				}
				
				predictedPositives++;
				
				Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = triplet.getKey() - 0.000000001;
				}
			}
			
			tp = numPositives[i];
			predictedPositives = data.n;
			Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives )); 
			
			if (maxFmeasure < Fmeasure ) {
				maxthreshold = minThreshold; //0.0;
				maxFmeasure = Fmeasure;
			}
				
			//logger.info(maxthreshold + " " + maxFmeasure);
			thresholds[i] = Math.min(0.5, maxthreshold);
			avgFmeasure += maxFmeasure;
		
			//logger.info("Label: " + i + " threshold: " + thresholds[i] + " F: " + maxFmeasure);
		}

//		for( int i=0; i < this.m; i++ )
//			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		
		this.numberOfPredictedPositives = numOfPositives;
		logger.info( "Validated macro F-measure: {}", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		this.validatedFmeasure = (avgFmeasure / (double) learner.getNumberOfLabels());

		
		logger.info( "\t --> !!!!!!!!!!!!! EUM fast end" );
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		logger.info("\t\t" + dateFormat.format(date));
		logger.info( "\t\tAvg. num. of predicted positives: " + numOfPositives / (double)(data.n) );
		logger.info( "############################################################" );		
		return thresholds;
	}


	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		logger.info("Tuning threshold (TTeumFast)...");
		
		logger.info( "\t --> @@@@@@@@@@@@@@@@@@@ EUM fast starts" );
		DateFormat dateFormat1 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date1 = new Date();
		logger.info("\t\t" + dateFormat1.format(date1));
		
		thresholds = new double[this.m];
		
		double avgFmeasure = 0.0;

		// for labels
		//int[] indices = new int[data.n];
		//for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		ArrayList<ComparableTriplet>[] posteriors = new ArrayList[this.m];
		int [] numPositives = new int[this.m];
		
		int numOfPositives = 0;
		
		for( int j = 0; j < data.n; j++ ) {
			
			HashSet<Integer> trueLabels = new HashSet<Integer>();
			
			for(int k = 0; k < data.y[j].length; k++) {
				trueLabels.add(data.y[j][k]);
			}
			
			HashSet<EstimatePair> sPE = new HashSet<EstimatePair>();
			// filter
			for(int i=0; i< sPEarray.x[j].length; i++ ) {
				if (sPEarray.x[j][i].value >this.minThreshold){
					sPE.add(new EstimatePair(sPEarray.x[j][i].index,sPEarray.x[j][i].value));
				}
			}
			
			
			numOfPositives += sPE.size();
			
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
				logger.info( "\t --> Instance: " + j +" (" + data.n + ")" );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (j+1) );
			}
			
		}
	
		for(int i = 0; i < posteriors.length; i++) {
		
			if(posteriors[i] == null) {
				thresholds[i] = this.minThreshold;
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
				
				//logger.info(triplet.getKey());
				
				if(triplet.gety() == 1) {
					tp++;
				}
				
				predictedPositives++;
				
				Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives ));
				
				if (maxFmeasure < Fmeasure ) {
					maxFmeasure = Fmeasure;
					maxthreshold = triplet.getKey() - 0.0000001;
				}
			}
			
			tp = numPositives[i];
			predictedPositives = data.n;
			Fmeasure = ((2.0*tp)) / ((double) ( numPositives[i] + predictedPositives )); 
			
			if (maxFmeasure < Fmeasure ) {
				maxthreshold = minThreshold; //0.0;
				maxFmeasure = Fmeasure;
			}
				
			//logger.info(maxthreshold + " " + maxFmeasure);
			thresholds[i] = Math.min(0.5, maxthreshold);
			avgFmeasure += maxFmeasure;
		
			//logger.info("Label: " + i + " threshold: " + thresholds[i] + " F: " + maxFmeasure);
		}

//		for( int i=0; i < this.m; i++ )
//			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		
		this.numberOfPredictedPositives = numOfPositives;
		
		logger.info( "Validated macro F-measure: {}", (avgFmeasure / (double) m) ) ;
		this.validatedFmeasure = (avgFmeasure / (double) m);

		
		logger.info( "\t --> !!!!!!!!!!!!! EUM fast end" );
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		logger.info("\t\t" + dateFormat.format(date));
		logger.info( "\t\tAvg. num. of predicted positives: " + numOfPositives / (double)(data.n) );
		logger.info( "############################################################" );		
		return thresholds;
	}
	
	
}

