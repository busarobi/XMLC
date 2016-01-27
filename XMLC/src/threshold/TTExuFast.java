package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTExuFast extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTExuFast.class);

	protected int epochs = 1;	
	protected int a = 1;
	protected int b = 1;

	protected int[] aInit = null;
	protected int[] bInit = null;
	
	
	public void setaInit(int[] aInit) {
		if (aInit.length != this.m ) {
			System.exit(-1);
		}
 
		this.aInit = aInit;
	}

	public void setbInit(int[] bInit) {
		if (bInit.length != this.m ) {
			System.exit(-1);
		}

		this.bInit = bInit;
	}
	
	public TTExuFast(int m, Properties properties) {
		super(m, properties);
		
		this.epochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.a = Integer.parseInt(properties.getProperty("a", "1") );
		this.b = Integer.parseInt(properties.getProperty("b", "100") );
		
		logger.info("#####################################################" );
		logger.info("#### EXU Fast" );
		logger.info("#### iter: " + this.epochs );
		logger.info("#### a: " + this.a );
		logger.info("#### b: " + this.b );
		logger.info("#####################################################" );		
	}

	
	
	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		{
			logger.info( "--> @@@@@@@@@@@@@@@ Start of EXU Fast" );
			logger.info( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.epochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));
		}
		
		int[] at = new int[this.m];
		int[] bt = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		if ( (this.a >= 0) && (this.aInit == null)) {
			for( int i = 0; i < this.m; i++ ) {

				at[i] = this.a;
				bt[i] = this.b;			
			}
		} else if  ( ( this.aInit != null ) && (this.bInit != null ) ) {
			for( int i = 0; i < this.m; i++ ) {
				at[i] = this.aInit[i];
				bt[i] = this.bInit[i];
			}			
		} else {
			logger.info("\t\t--> Initialized with the prior!");
			int[] numOfLabels = AVTable.getNumOfLabels(data);

			for( int i = 0; i < this.m; i++ ) {
				at[i] = numOfLabels[i];
				bt[i] = numOfLabels[i] + data.n;			
			}
		}
		
		for( int i = 0; i < this.m; i++ ) {		
			double F00 = (2.0 * at[i]) / ((double) bt[i]);
			double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
			double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);		
		
			this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
		}		
		
		
		learner.setThresholds(this.thresholds);
		int numOfPositives = 0;
		
		for( int e = 0; e < this.epochs; e++ ) { 
			
			for( int j = 0; j < data.n; j++ ) {

				HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();
				
				numOfPositives += predictedPositives.size();
				
				HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

				for(int predictedLabel : predictedPositives) {
					bt[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
				
				for(int m = 0; m < data.y[j].length; m++) {			
					int trueLabel = data.y[j][m];					
					bt[trueLabel]++;
					thresholdsToChange.add(trueLabel);
					if(predictedPositives.contains(trueLabel)) {
						at[trueLabel]++;
					}
				}

				for(int label: thresholdsToChange) {
					
					double F00 = (2.0 * at[label]) / ((double) bt[label]);
					double F01 = (2.0 * at[label]) / ((double) bt[label]+1);
					double F11 = (2.0 * (at[label]+1)) / ((double) bt[label]+2);
					
					double t = (F01 - F00) / (2*F01 - F00 - F11 );
					this.thresholds[label] = t;
					learner.setThreshold(label, t);
					
				}
				
				if ((j % 100000) == 0) {
					logger.info( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + (e+1)  + " (" + this.epochs + ")"  );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (j+1) );					
				}

				
			}

		}
		
		
		this.numberOfPredictedPositives = numOfPositives;
//		for( int i=0; i < this.m; i++ )
//			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		{
			logger.info( "--> !!!!!!!!!!!! End of EXU Fast" );
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));
			
			double avgFmeasure = 0.0;
			for( int i = 0; i < this.thresholds.length; i++ ){
				avgFmeasure += this.thresholds[i];
			}
		
			avgFmeasure = (2.0 * avgFmeasure) / (double)this.thresholds.length;
			this.validatedFmeasure = avgFmeasure;
			
			logger.info( "Validated macro F-measure: {}", avgFmeasure ) ;
			logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double)(data.n * this.epochs) );
			logger.info( "############################################################" );			
		}
		
		
		
		return thresholds;
		
	}



	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		{
			logger.info( "--> @@@@@@@@@@@@@@@ Start of EXU Fast" );
			logger.info( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.epochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));
		}
		
		int[] at = new int[this.m];
		int[] bt = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		if ( (this.a >= 0) && (this.aInit == null) ) {
			for( int i = 0; i < this.m; i++ ) {

				at[i] = this.a;
				bt[i] = this.b;			
			}
		} else if  ( ( this.aInit != null ) && (this.bInit != null ) ) {
			for( int i = 0; i < this.m; i++ ) {
				at[i] = this.aInit[i];
				bt[i] = this.bInit[i];
			}			
		} else {
			logger.info("\t\t--> Initialized with the prior!");
			int[] numOfLabels = AVTable.getNumOfLabels(data);

			for( int i = 0; i < this.m; i++ ) {
				at[i] = numOfLabels[i];
				bt[i] = numOfLabels[i] + data.n;			
			}
		}
		
		for( int i = 0; i < this.m; i++ ) {		
			double F00 = (2.0 * at[i]) / ((double) bt[i]);
			double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
			double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);		
		
			this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
		}		
		
		
		//learner.setThresholds(this.thresholds);
		int numOfPositives = 0;
		
		for( int e = 0; e < this.epochs; e++ ) { 
			
			for( int j = 0; j < data.n; j++ ) {

				//HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();
				HashSet<Integer> predictedPositives = new HashSet<Integer>(); 
				for(int i = 0; i < sPEarray.x[j].length; i++ ) {
					if (this.thresholds[sPEarray.x[j][i].index] < sPEarray.x[j][i].value) {
						predictedPositives.add(sPEarray.x[j][i].index);
					}
				}
				
				
				numOfPositives += predictedPositives.size();
				
				HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

				for(int predictedLabel : predictedPositives) {
					bt[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
				
				for(int m = 0; m < data.y[j].length; m++) {			
					int trueLabel = data.y[j][m];					
					bt[trueLabel]++;
					thresholdsToChange.add(trueLabel);
					if(predictedPositives.contains(trueLabel)) {
						at[trueLabel]++;
					}
				}

				for(int label: thresholdsToChange) {
					
					double F00 = (2.0 * at[label]) / ((double) bt[label]);
					double F01 = (2.0 * at[label]) / ((double) bt[label]+1);
					double F11 = (2.0 * (at[label]+1)) / ((double) bt[label]+2);
					
					double t = (F01 - F00) / (2*F01 - F00 - F11 );
					this.thresholds[label] = t;
					//learner.setThreshold(label, t);
					
				}
				
				if ((j % 100000) == 0) {
					logger.info( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + (e+1)  + " (" + this.epochs + ")"  );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (j+1) );					
				}

				
			}

		}
		
		
		this.numberOfPredictedPositives = numOfPositives;
//		for( int i=0; i < this.m; i++ )
//			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		{
			logger.info( "--> !!!!!!!!!!!! End of EXU Fast" );
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));
			
			double avgFmeasure = 0.0;
			for( int i = 0; i < this.thresholds.length; i++ ){
				avgFmeasure += this.thresholds[i];
			}
		
			avgFmeasure = (2.0 * avgFmeasure) / (double)this.thresholds.length;
			this.validatedFmeasure = avgFmeasure;
			
			logger.info( "Validated macro F-measure: {}", avgFmeasure ) ;
			logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double)(data.n * this.epochs) );
			logger.info( "############################################################" );			
		}
		
		
		
		return thresholds;
	}

}
