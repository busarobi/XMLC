package Learner;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.PriorityQueue;

import Data.AVTable;
import Data.ComparablePair;
import IO.DataReader;

public class FilterFastXML extends LearnerOutputPosteriors {

	public FilterFastXML(String fname) {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	public void outputPosteriors() throws Exception {		

		// compute threshold			
			int[] numOfPositivesTrain = AVTable.getNumOfLabels(this.traindata);
			int[] numOfPositivesValid = AVTable.getNumOfLabels(this.validdata);
			int[] numOfPositivesTest = AVTable.getNumOfLabels(this.testdata);
			
			System.out.println("Num of labels: " + this.traindata.m);
			double[] thresholds = new double[this.traindata.m];
			
			
			double N = (double) (this.traindata.n + this.validdata.n + this.testdata.n);
			
			for(int i = 0; i < this.traindata.m; i++ ) {
				int val = 0;
				if (i < numOfPositivesValid.length) {
					val = numOfPositivesValid[i];
				}
				
				int tst = 0;
				if (i < numOfPositivesTest.length) {
					tst = numOfPositivesTest[i];
				}
				
				double a = numOfPositivesTrain[i] + val + tst;
				thresholds[i] = Math.max(this.threshold, a / (a+N) ); 
			}
			
			
			DataReader testpostreader = new DataReader(this.posteriorFileTest, false, false);
			AVTable posteriors = testpostreader.read();
			String outfname = this.posteriorFileTest.replace(".txt", "_filtered.txt");
			System.out.println("Out: " + outfname );
			this.filterandwritePosteriorsToFile( thresholds, posteriors, outfname);

			
			
			DataReader validpostreader = new DataReader(this.posteriorFileValid, false, false);
			posteriors = validpostreader.read();
			outfname = this.posteriorFileValid.replace(".txt", "_filtered.txt");
			System.out.println("Out: " + outfname );
			this.filterandwritePosteriorsToFile( thresholds, posteriors, outfname);
	}

	public void filterandwritePosteriorsToFile( double[] thresholds, AVTable posteriors, String fname ) throws IOException{
		System.out.println("Output posteriors to " + fname );
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
		int numOfPositives = 0;
		for( int i = 0; i<posteriors.n; i++)
		{			
			for(int j=0; j<posteriors.x[i].length; j++ ) {
				if (posteriors.x[i][j].value > thresholds[posteriors.x[i][j].index+1] ){
					bf.write(  "" + (posteriors.x[i][j].index+1) + ":" + posteriors.x[i][j].value + " "  );
				}
			}
			
			bf.write( "\n" );			
		}
		
		bf.close();
		
	}
	
	public void readValidData() throws Exception {
		DataReader datareader = new DataReader(this.lableFileValid, false, false);
		validdata = datareader.read();
	}

//	public void readTrainData() throws Exception {
//		// reading train data
//		DataReader datareader = new DataReader(this.lableFileValid, false, false);
//		traindata = datareader.read();
////		if (fh != null ) {
////			traindata = fh.transformSparse(traindata);
////		}
//	}
//
	public void readTestData() throws Exception {
		// test
		DataReader testdatareader = new DataReader(this.labelFileTest,false, false );
		testdata = testdatareader.read();
//		if (fh != null ) {
//			testdata = fh.transformSparse(testdata);
//		}
	}

	
	public static void main(String[] args) throws Exception {

		FilterFastXML lm = new FilterFastXML(args[0]);
		
	    lm.readTrainData();    
	    lm.readValidData();
	    lm.readTestData();
	    
	    lm.outputLabels();
	    
	    lm.outputPosteriors();
	    

	}

}
