package IO;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import Learner.AbstractLearner;

public class Evaluator {

	private static Logger logger = LoggerFactory.getLogger(Evaluator.class);

	
    static public Map<String,Double> computePerformanceMetrics(AbstractLearner learner, AVTable data) {
		logger.info("--> Computing Hamming loss and F-measure...");

		double macroF = 0.0;
		int m = learner.getNumberOfLabels();
		
		int[] tp = new int[m];
		int[] yloc = new int[m];
		int[] haty = new int[m];
		
		double HL = 0.0;
		
		
		int numOfPositives = 0;
		
		for(int i = 0; i < data.n; i++ ) {
			

			HashSet<Integer> predictedLabels = learner.getPositiveLabels(data.x[i]);
			
			//logger.info("Predicted labels: " + predictedLabels.toString());
			
			int predpositloc = predictedLabels.size(); 
			numOfPositives += predpositloc;
			// Hamming
			int tploc = 0, fnloc = 0, fploc = 0;
						
			if ((data.y[i] != null) || (data.y[i].length >= 0) ) {				
				for(int trueLabel: data.y[i]) {
					if (trueLabel < m ) { // this label was seen in the training
						if(predictedLabels.contains(trueLabel)) {
							tploc++;
						} else {
							fnloc++;
						}
					}
				}			
			}
			fploc = predpositloc - tploc;
			HL += (fnloc + fploc);
			
			
			// F-score
			if ((data.y[i] != null) && (data.y[i].length > 0) ) {
				for(int trueLabel: data.y[i]) {
					if (trueLabel>= m) continue; // this label is not seen in the training
					
					if(predictedLabels.contains(trueLabel)) {
						tp[trueLabel]++;
					}
					yloc[trueLabel]++;
				}				
			} 
				

			for(int predictedLabel: predictedLabels) {
				haty[predictedLabel]++;
			}
			
			
			if ((i % 100000) == 0) {
				logger.info( "----->\t Evaluation Sample: "+ i +" (" + data.n + ")" );
				
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (i+1) );				
			}
			
		}
				
		HL = HL / ((double)data.n);
		double normalizedHL = (HL / (double)m);
		
		
		int presentedlabels = 0;
		for(int i = 0; i < m; i++) {
			double denum = (double) (yloc[i] + haty[i]);
			if (( denum>0.0) && (yloc[i]>0))
			{
				macroF += (2.0 * tp[i])/denum;
				presentedlabels++;
			}
		}
		
		double normalizedmacroF = macroF/(double) presentedlabels;
		
		TreeMap<String,Double> arr = new TreeMap<String,Double>();
		arr.put(" Hamming loss", HL);
		arr.put(" macro F-measure", macroF);
		arr.put( " learner.m", (double) m);
		arr.put( " Num of presented labels", (double) presentedlabels);
		

		arr.put(" Normalized macro F-measue (with presented labels)", normalizedmacroF);
		arr.put(" Normalized Hamming loss (with learner.m)", normalizedHL );

		
		return arr;

    }

    public static TreeMap<String,Double> computePrecisionAtk(AbstractLearner learner, AVTable data, int k) {
    	double[] precisionatK = new double[k];
    	
    	//int [] numLabels = new int [data.m];
    	
    	for(int i = 0; i < data.n; i++ ) {
    		TreeSet<EstimatePair> predictedLabels = learner.getTopKEstimates(data.x[i], k); //.getPositiveLabelsAndPosteriors(data.x[i]);
    		
    		
    		HashSet<Integer> trueLabels = new HashSet<Integer>();
			
			for(int m = 0; m < data.y[i].length; m++) {
				trueLabels.add(data.y[i][m]);
				//numLabels[data.y[i][m]]++;
			}
			
    		
    		//Hashtable<Integer,Integer> topklabel = new Hashtable<>();
    		//for( int j = 0; j < k; j++ ){
    		//	if ( predictedLabels.isEmpty() ) break;
    		//	ComparablePair p = predictedLabels.poll();
    		//	topklabel.put(p.getValue(), j);
    		//}
    		
    		//for( int j = 0; j < k; j++ ) iscorrectprediction[j] = 0;

			int[] iscorrectprediction = new int[k];
	    	int[] nunmofcorrectpredictionuptok = new int[k];
	    	
			int index = 0;
			while(!predictedLabels.isEmpty()) {
				
				EstimatePair eP = predictedLabels.pollFirst();
				
				int label = eP.getLabel();
				double p = eP.getP();
				
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

			if ((i % 100000) == 0) {
				logger.info( "----->\t Prec@ computation: "+ i +" (" + data.n + ")" );
				
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));								
			}
			
			
    	}
		
		
    	
		for( int j = 0; j < k; j++ ) {
			precisionatK[j] /= ((double) data.n);
		}
    	
    	
		//for(int i = 0; i < numLabels.length; i++) {
		//	logger.info("Label: " + i + " num: " + numLabels[i]);
		//}
		//logger.info("Num instances: " + data.n);
		
		TreeMap<String,Double> arr = new TreeMap<String,Double>();
		for(int i=0; i < k; i++){
			arr.put( "PrecAtK["+(i+1)+"]", precisionatK[i] );
		}
		
    	return arr;
    }

    /* 
    public static double[] computePrecisionAtk(AbstractLearner learner, AVTable data, int k) {
    	double[] precisionatK = new double[k];
    	int[] iscorrectprediction = new int[k];
    	int[] nunmofcorrectpredictionuptok = new int[k];
    	
    	for(int i = 0; i < data.n; i++ ) {
    		PriorityQueue<ComparablePair> predictedLabels = learner.getPositiveLabelsAndPosteriors(data.x[i]);
    		
    		Hashtable<Integer,Integer> topklabel = new Hashtable<>();
    		for( int j = 0; j < k; j++ ){
    			if ( predictedLabels.isEmpty() ) break;
    			ComparablePair p = predictedLabels.poll();
    			topklabel.put(p.getValue(), j);
    		}
    		
    		for( int j = 0; j < k; j++ ) iscorrectprediction[j] = 0;

			if ((data.y[i] != null) || (data.y[i].length >= 0) ) {				
				for(int trueLabel: data.y[i]) {
					Integer pos = topklabel.get(trueLabel);
					if (pos != null){
						iscorrectprediction[pos] = 1;
					}						
				}			
			}
    		
			nunmofcorrectpredictionuptok[0] = iscorrectprediction[0];
			for( int j = 1; j < k; j++ ) {
				nunmofcorrectpredictionuptok[j] = nunmofcorrectpredictionuptok[j-1] + iscorrectprediction[j];
			}
    		
			for( int j = 0; j < k; j++ ) {
				precisionatK[j] += (nunmofcorrectpredictionuptok[j] / ((double) (j+1)));
			}			
    	}
    	
    	
		for( int j = 0; j < k; j++ ) {
			precisionatK[j] /= ((double) data.n);
		}
    	
    	
    	return precisionatK;
    }
    */
    
    
//	public void updatePerformancesBasedOnInsatance( int labels, int[] forecast )
//	{
//		
//	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
