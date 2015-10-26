package IO;

import java.util.HashSet;

import Data.AVTable;
import Learner.SGDMLC;
import jsat.classifiers.BaseUpdateableClassifier;

public class PerformanceMeasures {
	
	public double computeHammingLoss(HashSet<Integer>[] predictedLabels, AVTable data ) {
		return computeHammingLoss( predictedLabels, data.y, data.n, data.m );
	}
		
	
	public double computeHammingLoss(HashSet<Integer>[] predictedLabels, int[][] y, int n, int m ) {
			
		double HL = 0.0;
			
		for(int i = 0; i < predictedLabels.length; i++ ) {
			
			int tp = 0, fn = 0;
			
			if (y[i] == null ) continue;
			
			for(int trueLabel: y[i]) {
				if(predictedLabels[i].contains(trueLabel)) {
					tp++;
				} else {
					fn++;
				}
			}
			
			HL += fn + (predictedLabels[i].size() - tp);

		}
		
		HL = HL / ((double)(n * m));
		
		return HL;
	}

	
	public double computeMacroF(HashSet<Integer>[] predictedLabels, AVTable data ) {
		return computeMacroF( predictedLabels, data.y, data.n, data.m );
	}
	
	
	public double computeMacroF(HashSet<Integer>[] predictedLabels,  int[][] y, int n, int m ) {
		
		double macroF = 0.0;
			
		int[] tp = new int[m];
		int[] yloc = new int[m];
		int[] haty = new int[m];
		
		for(int i = 0; i < predictedLabels.length; i++ ) {
			
			if (y[i] == null ) continue;
			
			for(int trueLabel: y[i]) {
				if(predictedLabels[i].contains(trueLabel)) {
					tp[trueLabel]++;
				}
				yloc[trueLabel]++;
			}

			for(int predictedLabel: predictedLabels[i]) {
				haty[predictedLabel]++;
			}
			
		}
		
		for(int i = 0; i < m; i++) {
			macroF += (2.0 * tp[i])/(double) (yloc[i] + haty[i]);
		}
		
		return macroF/(double) m;
	}

	
	
	
	public static void main(String[] args) throws Exception {
		
		String trainfileName = "../data/mediamill/train-exp1.svm";
		String testfileName = "../data/mediamill/test-exp1.svm";
		
		DataReader testdatareader = new DataReader( testfileName );
		AVTable testdata = testdatareader.read();
		
		HashSet<Integer>[] predictedLabels = new HashSet[testdata.n];
		for( int i = 0; i < testdata.n; i++ ) {
			
			predictedLabels[i] = new HashSet<Integer>();
			for(int trueLabel: testdata.y[i]) {
				
				predictedLabels[i].add(trueLabel);
			}
		}
	
		PerformanceMeasures  pm = new PerformanceMeasures();
		System.out.println( "Hamming loss: " + pm.computeHammingLoss(predictedLabels, testdata));
		System.out.println("Macro-F: " + pm.computeMacroF(predictedLabels, testdata));
		
	}
}
