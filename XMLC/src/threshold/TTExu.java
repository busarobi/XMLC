package threshold;

import java.util.HashSet;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTExu extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTExu.class);

	protected int epochs = 1;
	protected int initValueDenum = 1;
	
	public TTExu(int m, Properties properties) {
		super(m, properties);
		this.epochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.initValueDenum = Integer.parseInt(properties.getProperty("InitValueDenum", "1") );
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		int[] at = new int[this.m];
		int[] bt = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		
		for( int i = 0; i < this.m; i++ ) {
			//at[i] = (int) Math.round(prior[i] * 1000);
			//bt[i] = 1000;
			at[i] = 1;
			bt[i] = this.initValueDenum;
			
			double F00 = (2.0 * at[i]) / ((double) bt[i]);
			double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
			double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);
			
			
			this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
		}
		
		for( int e = 0; e < this.epochs; e++ ) { 
			for (int i = 0; i < this.m; i++) {

				// assume that the labels are ordered
				int currentLabel = 0;
				for (int j = 0; j < data.n; j++) {
					if ((indices[j] < data.y[j].length) && (data.y[j][indices[j]] == i)) {
						currentLabel = 1;
						indices[j]++;
					} else {
						currentLabel = 0;
					}

					double post = learner.getPosteriors(data.x[j], i);
					if (post > this.thresholds[i]) {
						bt[i]++;
						if (currentLabel == 1)
							at[i]++;
					}
					if (currentLabel == 1)
						bt[i]++;

					double F00 = (2.0 * at[i]) / ((double) bt[i]);
					double F01 = (2.0 * at[i]) / ((double) bt[i]+1);
					double F11 = (2.0 * (at[i]+1)) / ((double) bt[i]+2);
					
					
					this.thresholds[i] = (F01 - F00) / (2*F01 - F00 - F11 );
					//this.thresholds[i] = ((prior[i] * (F01 - F00)) + F00) / ((prior[i] * (2*F01 - F11 ) ) + F00 + F01 );
				}
												
			}

		}
		
		for( int i=0; i < this.m; i++ )
			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
		return thresholds;
		
	}

	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		// TODO Auto-generated method stub
		return null;
	}

}
