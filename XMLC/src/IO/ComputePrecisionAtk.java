package IO;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.EstimatePair;
import Data.Instance;
import Learner.AbstractLearner;

class ComputePrecisionAtk implements Runnable {
	private static Logger logger = LoggerFactory.getLogger(ComputePrecisionAtk.class);
	
	protected int k = 5;    	
	protected AbstractLearner learner = null;
	protected DataManager  data = null;
	protected double[] precisionatK = null;
	protected int numOfInstance = 0;

	public ComputePrecisionAtk(AbstractLearner learner, DataManager data, int k ) {
		this.learner = learner;
		this.k = k;    		
		this.data = data;
		this.precisionatK = new double[this.k];
		this.numOfInstance = 0;
	}

	@Override
	public void run() {
		this.numOfInstance = 0;
		//int [] numLabels = new int [data.m];
		while( data.hasNext() == true ) {    	
			Instance instance  = this.data.getNextInstance();
			this.numOfInstance++;

			TreeSet<EstimatePair> predictedLabels = this.learner.getTopKEstimates(instance.x, this.k); //.getPositiveLabelsAndPosteriors(data.x[i]);


			HashSet<Integer> trueLabels = new HashSet<Integer>();

			for(int m = 0; m < instance.y.length; m++) {
				trueLabels.add(instance.y[m]);
			}

			int[] iscorrectprediction = new int[k];
			int[] nunmofcorrectpredictionuptok = new int[k];

			int index = 0;
			while(!predictedLabels.isEmpty()) {

				EstimatePair eP = predictedLabels.pollFirst();

				int label = eP.getLabel();
				//double p = eP.getP();

				//logger.info(index + " label: " + label + " p: " + p);

				if(trueLabels.contains(label)) {
					iscorrectprediction[index]++;
				}

				index++;

			}

			nunmofcorrectpredictionuptok[0] = iscorrectprediction[0];
			for( int j = 1; j < k; j++ ) {
				nunmofcorrectpredictionuptok[j] = nunmofcorrectpredictionuptok[j-1] + iscorrectprediction[j];
			}

			for( int j = 0; j < k; j++ ) {
				precisionatK[j] += (nunmofcorrectpredictionuptok[j] / ((double) (j+1)));
			}			


			if ((numOfInstance % 100000) == 0) {
				logger.info( "----->\t Prec@ computation: "+ numOfInstance  );

				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));								
			}


		}


	}
}
