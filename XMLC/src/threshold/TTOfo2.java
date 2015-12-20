package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTOfo2 extends ThresholdTuning {
	protected int OFOepochs = 1;
	//protected int initValueDenum = 1;
	
	protected int a = 0;
	protected int b = 1;
	
	public TTOfo2(int m, Properties properties) {
		super(m, properties);
		
		this.OFOepochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.a = Integer.parseInt(properties.getProperty("a", "50") );
		this.b = Integer.parseInt(properties.getProperty("b", "1000") );
		
		System.out.println("#####################################################" );
		System.out.println("#### OFO2" );
		System.out.println("#### iter: " + this.OFOepochs );
		System.out.println("#### a: " + this.a );
		System.out.println("#### OFO2: " + this.b );
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

			for( int j = 0; j < data.n; j++ ) {

				HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();

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
			}

		}
		
		for( int i=0; i < this.m; i++ )
			System.out.println( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		{
			System.out.println( "############## End of TTOfo2" );
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			System.out.println("\t\t" + dateFormat.format(date));
			//System.out.printf( "Validated macro F-measure: %.5f\n", (avgFmeasure / (double) learner.getNumberOfLabels()) ) ;
		}

		
		return this.thresholds;
	}

}
