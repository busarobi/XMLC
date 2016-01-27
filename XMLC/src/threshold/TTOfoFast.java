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

public class TTOfoFast extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTOfoFast.class);

	protected int OFOepochs = 1;
	//protected int initValueDenum = 1;
	
	protected int a = 0;
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

	public TTOfoFast(int m, Properties properties) {
		super(m, properties);
		
		this.OFOepochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.a = Integer.parseInt(properties.getProperty("a", "1") );
		this.b = Integer.parseInt(properties.getProperty("b", "100") );
		
		logger.info("#####################################################" );
		logger.info("#### OFO Fast" );
		logger.info("#### iter: " + this.OFOepochs );
		logger.info("#### a: " + this.a );
		logger.info("#### b: " + this.b );
		logger.info("#####################################################" );		
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		
		{
			logger.info( "############################################################" );
			logger.info( "--> @@@@@@@@@@@@@ Start of TTOfoFast" );
			logger.info( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.OFOepochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));			
		}

		
		
		int[] a = new int[this.m];
		int[] b = new int[this.m];
					
		if ( (this.a >= 0 ) && (this.aInit == null)) { 
			for( int i = 0; i < this.m; i++ ) {
				a[i] = this.a;
				b[i] = this.b;
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}
		} else if  ( ( this.aInit != null ) && (this.bInit != null ) ) {
			for( int i = 0; i < this.m; i++ ) {
				a[i] = this.aInit[i];
				b[i] = this.bInit[i];
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}			
		} else if ( (this.a < 0) || (this.b < 0) ) {
			logger.info("\t\t--> Initialized with the prior!");
			int[] numOfLabels = AVTable.getNumOfLabels(data);

			for( int i = 0; i < this.m; i++ ) {
				a[i] = numOfLabels[i];
				b[i] = numOfLabels[i] + data.n;
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}
			
		}
		
		learner.setThresholds(this.thresholds);
		
		int numOfPositives = 0;		
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
			
			for( int j = 0; j < data.n; j++ ) {

				HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();

				numOfPositives += predictedPositives.size();
				
				HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

				for(int predictedLabel : predictedPositives) {
					b[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
								
				for(int m = 0; m < data.y[j].length; m++) {			
					int trueLabel = data.y[j][m];					
					b[trueLabel]++;
					thresholdsToChange.add(trueLabel);
					if(predictedPositives.contains(trueLabel)) {
						a[trueLabel]++;
					}
				}

				for(int label: thresholdsToChange) {
					double t = (double) a[label] / (double) b[label];
					learner.setThreshold(label, t); 
					this.thresholds[label] = t;
				}
				
				
				if ((j % 100000) == 0) {
					logger.info( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + (e+1)  + " (" + this.OFOepochs + ")"  );
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
			logger.info( "--> !!!!!!!!!!! End of TTOfoFast" );
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
			logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double)(data.n * this.OFOepochs) );
			logger.info( "############################################################" );			
		}

		
		return this.thresholds;
	}

	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		
		{
			logger.info( "############################################################" );
			logger.info( "--> @@@@@@@@@@@@@ Start of TTOfoFast" );
			logger.info( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.OFOepochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			logger.info("\t\t" + dateFormat.format(date));			
		}

		
		
		int[] a = new int[this.m];
		int[] b = new int[this.m];
					
		if ((this.a >= 0 ) && (this.aInit == null) ){ 
			for( int i = 0; i < this.m; i++ ) {
				a[i] = this.a;
				b[i] = this.b;
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}
		} else if  ( ( this.aInit != null ) && (this.bInit != null ) ) {
			for( int i = 0; i < this.m; i++ ) {
				a[i] = this.aInit[i];
				b[i] = this.bInit[i];
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}			
		} else {
			logger.info("\t\t--> Initialized with the prior!");
			int[] numOfLabels = AVTable.getNumOfLabels(data);

			for( int i = 0; i < this.m; i++ ) {
				a[i] = numOfLabels[i];
				b[i] = numOfLabels[i] + data.n;
						
				this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
			}
			
		}
		
		//learner.setThresholds(this.thresholds);
		
		int numOfPositives = 0;		
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
			
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
					b[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
								
				for(int m = 0; m < data.y[j].length; m++) {			
					int trueLabel = data.y[j][m];					
					b[trueLabel]++;
					thresholdsToChange.add(trueLabel);
					if(predictedPositives.contains(trueLabel)) {
						a[trueLabel]++;
					}
				}

				for(int label: thresholdsToChange) {
					double t = (double) a[label] / (double) b[label];
					//learner.setThreshold(label, t); 
					this.thresholds[label] = t;
				}
				
				
				if ((j % 100000) == 0) {
					logger.info( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + (e+1)  + " (" + this.OFOepochs + ")"  );
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
			logger.info( "--> !!!!!!!!!!! End of TTOfoFast" );
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
			logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double)(data.n * this.OFOepochs) );
			logger.info( "############################################################" );			
		}

		
		return this.thresholds;
	}

}
