package threshold;

import java.util.HashSet;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class TTOfo extends ThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(TTOfo.class);

	protected int OFOepochs = 1;
	protected int initValueDenum = 1;
	
	public TTOfo(int m, Properties properties) {
		super(m, properties);
		
		this.OFOepochs = Integer.parseInt(properties.getProperty("ThresholdEpochs", "1") );
		this.initValueDenum = Integer.parseInt(properties.getProperty("InitValueDenum", "1") );
	}

	@Override
	public double[] validate(AVTable data, AbstractLearner learner) {
		int[] TP = new int[this.m];
		int[] P = new int[this.m];
		int[] PredP = new int[this.m];
		
		int[] indices = new int[data.n];
		for( int i = 0; i < data.n; i++ ) indices[i] = 0;
		
		//double[] prior = AVTable.getPrior(data);
		
		for( int i = 0; i < this.m; i++ ) {
			TP[i] = 1;
			P[i] = this.initValueDenum;
			PredP[i] = this.initValueDenum;
						
			this.thresholds[i] = ((double) TP[i]) / ((double) P[i] + PredP[i]);
		}
		
		for( int e = 0; e < this.OFOepochs; e++ ) { 
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
						PredP[i]++;
						if (currentLabel == 1)
							TP[i]++;
					}
					if (currentLabel == 1)
						P[i]++;

					this.thresholds[i] = ((double) TP[i]) / ((double) P[i] + PredP[i]);
				}
												
			}

		}
		
		for( int i=0; i < this.m; i++ )
			logger.info( "Class: " + i + " Th: " + String.format("%.4f", this.thresholds[i])  );
	
		return this.thresholds;
	}

	@Override
	public double[] validate( AVTable data, AVTable sPEarray ) {
		// TODO Auto-generated method stub
		return null;
	}

}
