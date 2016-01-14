package threshold;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Properties;

import Data.AVTable;
import Learner.AbstractLearner;

public class TTExuFast extends ThresholdTuning {

	protected int epochs = 1;	
	protected int a = 1;
	protected int b = 1;
	
	public TTExuFast(int m, Properties properties) {
		super(m, properties);
		
		this.epochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.a = Integer.parseInt(properties.getProperty("a", "1") );
		this.b = Integer.parseInt(properties.getProperty("b", "100") );
		
		System.out.println("#####################################################" );
		System.out.println("#### EXU Fast" );
		System.out.println("#### iter: " + this.epochs );
		System.out.println("#### a: " + this.a );
		System.out.println("#### b: " + this.b );
		System.out.println("#####################################################" );		
	}

	
	
	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		{
			System.out.println( "############## Start of EXU Fast" );
			System.out.println( "Initial a:" +  this.a + "\tInitial b: " + this.b + "\tNumber of epochs: " + this.epochs);
			DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
			Date date = new Date();
			System.out.println("\t\t" + dateFormat.format(date));
		}
		
		int[] at = new int[this.m];
		int[] bt = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		
		for( int i = 0; i < this.m; i++ ) {
			//at[i] = (int) Math.round(prior[i] * 1000);
			//bt[i] = 1000;
			at[i] = a;
			bt[i] = b;
			
			double F00 = (2.0 * at[i]) / ((double) bt[i]);
			double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
			double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);
			
			
			this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
		}
		
		
		learner.setThresholds(this.thresholds);
		
		for( int e = 0; e < this.epochs; e++ ) { 

			int numOfPositives = 0;
			for( int j = 0; j < data.n; j++ ) {

				HashSet<Integer> predictedPositives = learner.getPositiveLabels(data.x[j]); //.getSparseProbabilityEstimates();
				
				numOfPositives += predictedPositives.size();
				
				HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

				for(int predictedLabel : predictedPositives) {
					bt[predictedLabel]++;
					thresholdsToChange.add(predictedLabel);
				}
				
				HashSet<Integer> trueLabels = new HashSet<Integer>();
			
				for(int m = 0; m < data.y[j].length; m++) {
					trueLabels.add(data.y[j][m]);
				}
				
				for(int trueLabel : trueLabels) {
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
					System.out.println( "\t --> Instance: " + j +" (" + data.n + "), epoch: " + e  + "(" + this.epochs + ")"  );
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
			System.out.println( "############## End of EXU Fast" );
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
		
		
		
		return thresholds;
		
	}

}
