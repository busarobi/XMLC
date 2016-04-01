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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparablePair;

public class LearnerOutputPosteriorsMeasureTime extends LearnerOutputPosteriors {
	private static Logger logger = LoggerFactory.getLogger(LearnerOutputPosteriorsMeasureTime.class);
	
	public LearnerOutputPosteriorsMeasureTime(String fname) {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	public void outputPosteriors() throws Exception {		
		
		this.readTrainData();
		int[] numOfPositivesTrain = AVTable.getNumOfLabels(this.traindata);
		int[] numOfPositivesValid = AVTable.getNumOfLabels(this.validdata);
		int[] numOfPositivesTest = AVTable.getNumOfLabels(this.testdata);
		double[] thresholds = new double[this.validdata.m];
		
		
		double N = (double) (this.traindata.n + this.validdata.n + this.testdata.n);
		
		for(int i = 0; i < this.traindata.m; i++ ) {
			double a = numOfPositivesTrain[i] + numOfPositivesValid[i] + numOfPositivesTest[i];
			thresholds[i] = Math.max(this.threshold, a / (a+N) ); 
		}
		
		this.learner.setThresholds(thresholds);
	
		//this.computePosteriorsToFile(this.learner, this.validdata);
		
		long tStart = System.currentTimeMillis();
		//this.learner.innerProdCounter = 0;
		this.computePosteriorsToFile(this.learner, this.testdata);
		long tEnd = System.currentTimeMillis();	
		long tDelta = tEnd - tStart;
		double elapsedSeconds = tDelta / 1000.0;
		//System.out.println("Elapsed seconds: " + elapsedSeconds );
		logger.info("Elapsed seconds: " + elapsedSeconds);
		logger.info("Test instances: " + this.testdata.n );
		//logger.info("Inner product: " + this.learner.innerProdCounter );
		//System.out.println("Elapsed seconds per instance: " + elapsedSeconds / ((double)this.testdata.n ) );
		logger.info("Elapsed seconds per instance: " + elapsedSeconds / ((double)this.testdata.n ));
		//logger.info("Inner product per instance: " + this.learner.innerProdCounter / ((double)this.testdata.n ) );
		
		
	}
	
	
	
	public void computePosteriorsToFile( AbstractLearner learner, AVTable data ) throws IOException{
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		logger.info("############ START: " + dateFormat.format(date));
		
		int numOfPositives = 0;
		for( int i = 0; i<data.n; i++)
		{
			PriorityQueue<ComparablePair> sPE = learner.getPositiveLabelsAndPosteriors(data.x[i]);
			//HashSet<EstimatePair> sPE = learner.getSparseProbabilityEstimates(data.x[i], minThreshold);
			numOfPositives += sPE.size();
			
			List<ComparablePair> sortedList = new ArrayList<>();
			sortedList.addAll(sPE);
			Collections.sort(sortedList, new Comparator<ComparablePair>() {
				@Override
				public int compare(ComparablePair o1, ComparablePair o2) {
					return Integer.compare(o1.getValue(), o2.getValue());
				}
			});
			
//			for(ComparablePair pred : sPE) {
//				bf.write(  "" + (pred.getValue()+1) + ":" + pred.getKey() + " "  );
//			}
//			
//			bf.write( "\n" );
			
			if ((i % 100000) == 0) {
				logger.info( "\t --> Instance: " + i +" (" + data.n + ")" );
				dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (i+1) );
			}
			
		}

		
		dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		date = new Date();
		logger.info("############ END: " + dateFormat.format(date));
		
		
	}
	
	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		LearnerOutputPosteriorsMeasureTime lm = new LearnerOutputPosteriorsMeasureTime(args[0]);
				
	    lm.train();	    
	    lm.readValidData();
	    lm.readTestData();

	    
//	    lm.writeMinThresholdToFile( "/Users/busarobi/work/XMLC/Posteriors/WikiLSHTC/min_threshold.txt" );
	    
	    //lm.compositeEvaluation();
	    
//	    lm.outputLabels();
	    
	    lm.outputPosteriors();
	    
	}
	
}
