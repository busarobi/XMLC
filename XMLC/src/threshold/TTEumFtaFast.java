package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparableTriplet;
import Data.EstimatePair;
import IO.Evaluator;
import Learner.AbstractLearner;

public class TTEumFtaFast extends TTEumFast {
	private static Logger logger = LoggerFactory.getLogger(TTEumFtaFast.class);
	
	private double[] setOfThresholds = null;
	
	
	public TTEumFtaFast(int m, Properties properties, double[] ths ) {
		super(m, properties);
		
		this.setOfThresholds = ths;
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		return null;
	}	
	
	
	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		logger.info("Tuning threshold (TTEumFtaFast)...");
		
		logger.info( "\t --> @@@@@@@@@@@@@@@@@@@ EUMFTA fast starts" );
		DateFormat dateFormat1 = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date1 = new Date();
		logger.info("\t\t" + dateFormat1.format(date1));
		
		thresholds = new double[this.m];		
		double[] Fscores = new double[this.m];
		for( int i = 0; i < this.m; i++ ) { 
			Fscores[i] = 0.0;
			thresholds[i] = this.setOfThresholds[0];
		}
		
		double[] tmpThresholds= new double[this.m];
		
		for( int ti = 0; ti < this.setOfThresholds.length; ti++ ){			
			for( int j = 0; j < this.m; j++ ) tmpThresholds[j] = this.setOfThresholds[ti];
			
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(data, sPEarray, tmpThresholds );
			// compute F-measure
			double[] perf = Evaluator.computeFscores(positiveLabelsArray, data );
			
			for( int i = 0; i < this.m; i++ ) {
				if ( perf[i] > Fscores[i] ) {
					Fscores[i] = perf[i];
					thresholds[i] = this.setOfThresholds[ti];
				}
			}
		}
		
		
		this.validatedFmeasure = 0.0;
		for( int i = 0; i < this.m; i++ ) {
			this.validatedFmeasure += Fscores[i];
		}
		this.validatedFmeasure /= ((double)this.m);
		
		logger.info( "\t --> !!!!!!!!!!!!! EUMFTA fast end" );
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		logger.info("\t\t" + dateFormat.format(date));
//		logger.info( "\t\tAvg. num. of predicted positives: " + numOfPositives / (double)(data.n) );
		logger.info( "############################################################" );		
		return thresholds;
	}
	
	protected HashSet<Integer>[] getPositiveLabels(AVTable labels, AVTable posteriors, double[] thresholds ){
		HashSet<Integer>[] positiveLabelsArray = new HashSet[labels.n];
		for(int i=0; i<labels.n; i++ ){
			positiveLabelsArray[i] = new HashSet<Integer>();
			for(int j=0; j < posteriors.x[i].length; j++) {
				int labelidx = posteriors.x[i][j].index;
				double post = posteriors.x[i][j].value;
				if ( post > thresholds[labelidx] ) {
					positiveLabelsArray[i].add(labelidx);
				}
			}
		}
		return positiveLabelsArray;
	}
	
	
}
