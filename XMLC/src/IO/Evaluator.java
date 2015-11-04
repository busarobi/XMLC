package IO;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import Data.AVTable;
import Learner.AbstractLearner;

public class Evaluator {

	
	
    static public Map<String,Double> computePerformanceMetrics(AbstractLearner learner, AVTable data) {
		System.out.println("--> Computing Hamming loss and F-measure...");

		double macroF = 0.0;
		int m = learner.getNumberOfLabels();
		
		int[] tp = new int[m];
		int[] yloc = new int[m];
		int[] haty = new int[m];
		
		double HL = 0.0;
		
		
		int numOfPositives = 0;
		
		for(int i = 0; i < data.n; i++ ) {
			
			
			
			if ((data.y[i] == null) || (data.y[i].length == 0) ) continue;
			
			
			HashSet<Integer> predictedLabels = learner.getPositiveLabels(data.x[i]);
			numOfPositives += predictedLabels.size();
			
			
			// Hamming
			int tploc = 0, fn = 0;
			int tmpsize = 0;
			for(int trueLabel: data.y[i]) {
				if (trueLabel < m ) { // this label was seen in the training
					tmpsize++;
					if(predictedLabels.contains(trueLabel)) {
						tploc++;
					} else {
						fn++;
					}
				}
			}
			
			HL += (fn + (tmpsize - tploc));

			
			
			// F-score
			for(int trueLabel: data.y[i]) {
				if (trueLabel>= m) continue; // this label is not seen in the training
				
				if(predictedLabels.contains(trueLabel)) {
					tp[trueLabel]++;
				}
				yloc[trueLabel]++;
			}

			for(int predictedLabel: predictedLabels) {
				haty[predictedLabel]++;
			}
			
			
			if ((i % 10000) == 0) {
				System.out.println( "----->\tSample: "+ i +" (" + data.n + ")" );
				
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				System.out.println("\t\t" + dateFormat.format(date));
				System.out.println( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (i+1) );				
			}
			
		}
		
		HL = HL / ((double)(data.n * m));
		
		int presentedlabels = 0;
		for(int i = 0; i < m; i++) {
			double denum = (double) (yloc[i] + haty[i]);
			if (( denum>0.0) && (yloc[i]>0))
			{
				macroF += (2.0 * tp[i])/denum;
				presentedlabels++;
			}
		}
		
		macroF = macroF/(double) presentedlabels;
		
		Map<String,Double> arr = new HashMap<String,Double>();
		arr.put("Hamming loss", HL);
		arr.put("macro F-measure", macroF);
		
		return arr;

    }
	
	public void updatePerformancesBasedOnInsatance( int labels, int[] forecast )
	{
		
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
