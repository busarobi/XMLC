package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTOfoFast extends ThresholdTuning {
	protected int OFOepochs = 1;
	//protected int initValueDenum = 1;
	
	protected int a = 0;
	protected int b = 1;
	
	public TTOfoFast(int m, Properties properties) {
		super(m, properties);
		
		this.OFOepochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.a = Integer.parseInt(properties.getProperty("a", "1") );
		this.b = Integer.parseInt(properties.getProperty("b", "100") );
		
		System.out.println("#####################################################" );
		System.out.println("#### OFO2" );
		System.out.println("#### iter: " + this.OFOepochs );
		System.out.println("#### a: " + this.a );
		System.out.println("#### b: " + this.b );
		System.out.println("#####################################################" );		
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		
		{
			System.out.println( "############## Start of TTOfo2" );
			System.out.println( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.OFOepochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			System.out.println("\t\t" + dateFormat.format(date));
			System.out.println( "############################################################" );
		}

		
		
		int[] a = new int[this.m];
		int[] b = new int[this.m];
					
		for( int i = 0; i < this.m; i++ ) {
			a[i] = this.a;
			b[i] = this.b;
						
			this.thresholds[i] = ((double) a[i]) / ((double) b[i]);
		}
		
		learner.setThresholds(this.thresholds);
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
			
			int numOfPositives = 0;
			
			for( int j = 0; j < data.n; j++ ) {

				HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();

				numOfPositives += predictedPositives.size();
				
				HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

				for(int predictedLabel : predictedPositives) {
					b[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
				
				HashSet<Integer> trueLabels = new HashSet<Integer>();
			
				for(int m = 0; m < data.y[j].length; m++) {
					trueLabels.add(data.y[j][m]);
				}
				
				for(int trueLabel : trueLabels) {
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
					System.out.println( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + e  + "(" + this.OFOepochs + ")"  );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					System.out.println( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (j+1) );
				}
				
			}

		}
		
//		for( int i=0; i < this.m; i++ )
//			System.out.println( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		{
			System.out.println( "############## End of TTOfo2" );
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			System.out.println("\t\t" + dateFormat.format(date));
			
			double avgFmeasure = 0.0;
			for( int i = 0; i < this.thresholds.length; i++ ){
				avgFmeasure += this.thresholds[i];
			}
			
			avgFmeasure = (2.0 * avgFmeasure) / (double)this.thresholds.length;			
			
			System.out.printf( "Validated macro F-measure: %.5f\n", avgFmeasure ) ;			
			System.out.println( "############################################################" );			
		}

		
		return this.thresholds;
	}

}
