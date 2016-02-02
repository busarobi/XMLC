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
		//int m = learner.getNumberOfLabels();
		int m = data.m;
		
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
		int presentedOrForecasted = 0;		
		for(int i = 0; i < data.m; i++) {			
			int denum =  (yloc[i] + haty[i]);
			
			if (yloc[i]>0)
				presentedlabels++;
			
			if ( denum == 0) 
			{
				macroF += 1.0; // 0.0 / 0.0 = 1
			} else {
				macroF += (2.0 * tp[i])/((double)denum);
				presentedOrForecasted++;								
			}
		}
						
		double normalizedmacroF = macroF/data.m;
		
		TreeMap<String,Double> arr = new TreeMap<String,Double>();
		arr.put(" Hamming loss", HL);
		arr.put(" macro F-measure", macroF);
		arr.put(" Presented Label", macroF);
		
		arr.put( " m (d1)", (double) data.m);
		arr.put( " presented (d2)", (double) presentedlabels);
		arr.put( " presented and forecasted (d3)", (double) presentedOrForecasted);
		
		arr.put( " unormalized Fscore with 1 (e1)", macroF);
		arr.put( " unormalized Fscore with 0 (e2)", macroF-presentedOrForecasted );
		
		arr.put( " e1/d1", macroF / data.m );
		arr.put( " e1/d2", macroF / presentedlabels );
		arr.put( " e1/d3", macroF / presentedOrForecasted );

		arr.put( " e2/d1", ( macroF - presentedOrForecasted) / data.m );
		arr.put( " e2/d2", ( macroF - presentedOrForecasted) / presentedlabels );
		arr.put( " e2/d3", ( macroF - presentedOrForecasted) / presentedOrForecasted );
		
				

		arr.put(" Normalized macro F-measue (with m)", normalizedmacroF);
		arr.put(" Normalized Hamming loss (with m)", normalizedHL );

		
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
	

	
    public static Map<String,Double> computePerformanceMetrics(HashSet<Integer>[] positiveLabelsArray, AVTable data) {
		logger.info("--> Computing Hamming loss and F-measure...");

		double macroF = 0.0;
		
		
		int[] tp = new int[data.m];
		int[] yloc = new int[data.m];
		int[] haty = new int[data.m];
		
		double HL = 0.0;
		
		
		int numOfPositives = 0;
		
		for(int i = 0; i < data.n; i++ ) {
			
			HashSet<Integer> predictedLabels = positiveLabelsArray[i];
			//HashSet<Integer> predictedLabels = learner.getPositiveLabels(data.x[i]);
			
			//logger.info("Predicted labels: " + predictedLabels.toString());
			
			int predpositloc = predictedLabels.size(); 
			numOfPositives += predpositloc;
			// Hamming
			int tploc = 0, fnloc = 0, fploc = 0;
						
			if ((data.y[i] != null) || (data.y[i].length >= 0) ) {				
				for(int trueLabel: data.y[i]) {
					if (trueLabel < data.m ) { // this label was seen in the training
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
					if (trueLabel>= data.m) continue; // this label is not seen in the training
					
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
		double normalizedHL = (HL / (double)data.m);
		double macroF0 = 0.0;

		int presentedlabels = 0;
		int presentedOrForecasted = 0;		
		for(int i = 0; i < data.m; i++) {			
			int denum =  (yloc[i] + haty[i]);
			
			if (yloc[i]>0)
				presentedlabels++;
			
			if ( denum == 0) 
			{
				macroF += 1.0; // 0.0 / 0.0 = 1
			} else {
				macroF += (2.0 * tp[i])/((double)denum);
				macroF0 += (2.0 * tp[i])/((double)denum);
				presentedOrForecasted++;								
			}
		}
						
		double normalizedmacroF = macroF/data.m;
		
		
		TreeMap<String,Double> arr = new TreeMap<String,Double>();
		arr.put(" Hamming loss", HL);
		arr.put(" macro F-measure", macroF);
		//arr.put(" Presented Label", macroF);
		
		arr.put( " m (d1)", (double) data.m);
		arr.put( " presented (d2)", (double) presentedlabels);
		arr.put( " presented and forecasted (d3)", (double) presentedOrForecasted);
		
		arr.put( " unormalized Fscore with 1 (e1)", macroF);
		arr.put( " unormalized Fscore with 0 (e2)", macroF0 );
		
		arr.put( " e1/d1", macroF / data.m );
		arr.put( " e1/d2", macroF / presentedlabels );
		arr.put( " e1/d3", macroF / presentedOrForecasted );

		arr.put( " e2/d1", macroF0  / data.m );
		arr.put( " e2/d2", macroF0 / presentedlabels );
		arr.put( " e2/d3", macroF0 / presentedOrForecasted );
		
		
		//arr.put( " Num of presented labels", (double) presentedlabels);
		

		arr.put(" Normalized macro F-measue (with m)", normalizedmacroF);
		arr.put(" Normalized Hamming loss (with m)", normalizedHL );
		arr.put(" num. of predicted positives", (double)numOfPositives );
		arr.put(" avg. num. of predicted positives", (double)numOfPositives/ (double) data.n );
		
		return arr;

    }
    
    public static double[] computeFscores(HashSet<Integer>[] positiveLabelsArray, AVTable data) {
		logger.info("--> Computing Hamming loss and F-measure...");

		double[] Fscores = new double[data.m];
		
		
		int[] tp = new int[data.m];
		int[] yloc = new int[data.m];
		int[] haty = new int[data.m];
		
		int numOfPositives = 0;
		
		for(int i = 0; i < data.n; i++ ) {
			
			HashSet<Integer> predictedLabels = positiveLabelsArray[i];
			//HashSet<Integer> predictedLabels = learner.getPositiveLabels(data.x[i]);
			
			int predpositloc = predictedLabels.size(); 
			numOfPositives += predpositloc;
			
			
			// F-score
			if ((data.y[i] != null) && (data.y[i].length > 0) ) {
				for(int trueLabel: data.y[i]) {
					if (trueLabel>= data.m) continue; // this label is not seen in the training
					
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
				
		int presentedlabels = 0;
		for(int i = 0; i < data.m; i++) {
			int denum =  (yloc[i] + haty[i]);
			if ( denum == 0) 
			{
				Fscores[i] = 1.0; // 0.0 / 0.0 = 1
			} else {
				Fscores[i] = (2.0 * tp[i])/((double)denum);
				presentedlabels++;				
			}
		}				
		
		return Fscores;

    }
    
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
