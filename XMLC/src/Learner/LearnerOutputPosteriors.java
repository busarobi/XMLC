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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparablePair;
import Data.EstimatePair;
import IO.DataReader;
import IO.Evaluator;

public class LearnerOutputPosteriors extends LearnerManager {
	private static Logger logger = LoggerFactory.getLogger(LearnerOutputPosteriors.class);

	protected String posteriorFileValid = null;
	protected String posteriorFileTest = null;
	protected String lableFileValid = null;
	protected String labelFileTest = null;
	
	protected double threshold = 0.01;
	
	public LearnerOutputPosteriors(String fname) {
		super(fname);
		
		logger.info("#####################################################" );
		logger.info("#### LearnerOutputPosteriors" );
		
		if (!this.properties.containsKey("ValidPostFile")) {
			logger.info("No valid posterior file is given!");
			System.exit(-1);
		}
		
		this.posteriorFileValid = this.properties.getProperty("ValidPostFile");		
		logger.info("### Valid posterior file: " + this.posteriorFileValid);
		
		if (!this.properties.containsKey("TestPostFile")) {
			logger.info("No test posterior file is given!");
			System.exit(-1);
		}
		
		this.posteriorFileTest = this.properties.getProperty("TestPostFile");
		logger.info("### Test posteriors file: " + this.posteriorFileTest );
		
		if (!this.properties.containsKey("ValidLabelFile")) {
			logger.info("No valid posterior file is given!");
			System.exit(-1);
		}
		
		this.lableFileValid = this.properties.getProperty("ValidLabelFile");
		logger.info("### Valid label file: " + this.lableFileValid );
		
		if (!this.properties.containsKey("TestLabelFile")) {
			logger.info("No test label file is given!");
			System.exit(-1);
		}
			
		this.labelFileTest = this.properties.getProperty("TestLabelFile");
		logger.info("### Test label file: " + this.labelFileTest );

		this.threshold = Double.parseDouble(properties.getProperty("minThreshold", "0.001") );
		
		logger.info("#### Min threshold: " + this.threshold );
		logger.info("#####################################################" );		
		
		
	}

	public void compositeEvaluation()
	{
		Map<String,Double> perfvalidpreck = Evaluator.computePrecisionAtk(learner, validdata, 5);
		Map<String,Double> perftestpreck = Evaluator.computePrecisionAtk(learner, testdata, 5);
		
		logger.info( "#### Valid:" );
		
		for ( String perfName : perfvalidpreck.keySet() ) {
			logger.info("##### Valid " + perfName + ": "  + perfvalidpreck.get(perfName) );			
		}
		
		
		logger.info( "#### Test:" );
				
		
		for ( String perfName : perftestpreck.keySet() ) {
			logger.info("##### Test " + perfName + ": "  + perftestpreck.get(perfName));
			
		}				
	}
	
	public void outputPosteriors() throws Exception {		
		
//		if (this.threshold < 0.0) {
			this.readTrainData();
			int[] numOfPositivesTrain = AVTable.getNumOfLabels(this.traindata);
			int[] numOfPositivesValid = AVTable.getNumOfLabels(this.validdata);
			int[] numOfPositivesTest = AVTable.getNumOfLabels(this.testdata);
			double[] thresholds = new double[this.validdata.m];
			
			
			double N = (double) (this.traindata.n + this.validdata.n + this.testdata.n);
			
//			double avgThreshold = 0.0;
//			for(int i = 0; i < this.traindata.m; i++ ) {
//				double a = numOfPositivesTrain[i] + numOfPositivesValid[i] + numOfPositivesTest[i];
//				thresholds[i] = a / (a+N);
//				avgThreshold += thresholds[i];
//			}
//			avgThreshold /= N;
//			logger.info("Avg. threshold = :" + avgThreshold );

			for(int i = 0; i < this.traindata.m; i++ ) {
				double a = numOfPositivesTrain[i] + numOfPositivesValid[i] + numOfPositivesTest[i];
				thresholds[i] = Math.max(this.threshold, a / (a+N) ); 
			}
			
			this.learner.setThresholds(thresholds);
		
			this.writePosteriorsToFile(this.learner, this.validdata, this.posteriorFileValid);
			this.writePosteriorsToFile(this.learner, this.testdata, this.posteriorFileTest);
//		} else {
//			this.writePosteriorsToFile(this.learner, this.validdata, this.posteriorFileValid, this.threshold);
//			this.writePosteriorsToFile(this.learner, this.testdata, this.posteriorFileTest, this.threshold);
//		}
	}

	
	public void writeMinThresholdToFile( String fname ) throws Exception{
		
		this.readTrainData();
		this.readTestData();
		
		int[] numOfPositivesTrain = AVTable.getNumOfLabels(this.traindata);		
		int[] numOfPositivesTest = AVTable.getNumOfLabels(this.testdata);
    
		double N = (double) (this.traindata.n + this.testdata.n);
		double[] thresholds = new double[this.testdata.m];
		double avgThreshold = 0.0;
		for(int i = 0; i < this.traindata.m; i++ ) {
			double a = numOfPositivesTrain[i] + numOfPositivesTest[i];
			thresholds[i] = a / (a+N);
			avgThreshold += thresholds[i];
		}
		avgThreshold /= N;
		logger.info("Avg. threshold = :" + avgThreshold );
		
		logger.info("Output labels to " + fname );
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
		
		for( int i = 0; i<thresholds.length; i++)
		{		
			double a = numOfPositivesTrain[i] + numOfPositivesTest[i];
			bf.write(  "" + ((int)a) + "," + thresholds[i] );
			bf.write( "\n" );
		}
		
		bf.close();		
	}
	
	
	
	public void writePosteriorsToFile( AbstractLearner learner, AVTable data, String fname ) throws IOException{
		logger.info("Output posteriors to " + fname );
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
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
			
			for(ComparablePair pred : sPE) {
				bf.write(  "" + (pred.getValue()+1) + ":" + pred.getKey() + " "  );
			}
			
			bf.write( "\n" );
			
			if ((i % 100000) == 0) {
				logger.info( "\t --> Instance: " + i +" (" + data.n + ")" );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (i+1) );
			}
			
		}
		
		bf.close();
		
	}
	
	
	
	
	public void writePosteriorsToFile( AbstractLearner learner, AVTable data, String fname, double minThreshold ) throws IOException{
		logger.info("Output posteriors to " + fname );
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
		int numOfPositives = 0;
		for( int i = 0; i<data.n; i++)
		{
			HashSet<EstimatePair> sPE = learner.getSparseProbabilityEstimates(data.x[i], minThreshold);
			numOfPositives += sPE.size();
			
			List<EstimatePair> sortedList = new ArrayList<>();
			sortedList.addAll(sPE);
			Collections.sort(sortedList, new Comparator<EstimatePair>() {
				@Override
				public int compare(EstimatePair o1, EstimatePair o2) {
					return Integer.compare(o1.getLabel(), o2.getLabel());
				}
			});
			
			for(EstimatePair pred : sPE) {
				bf.write(  "" + (pred.getLabel()+1) + ":" + pred.getP() + " "  );
			}
			
			bf.write( "\n" );
			
			if ((i % 100000) == 0) {
				logger.info( "\t --> Instance: " + i +" (" + data.n + ")" );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				logger.info( "\t\t Avg. num. of predicted positives: " + numOfPositives / (double) (i+1) );
			}
			
		}
		
		bf.close();
		
	}
	
	public void outputLabels() throws IOException {
		logger.info("Output valid label" );
		DataReader.writeLabels( this.lableFileValid, this.validdata );
		logger.info("Output test label" );
		DataReader.writeLabels( this.labelFileTest, this.testdata );
	}
	
	
	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		LearnerOutputPosteriors lm = new LearnerOutputPosteriors(args[0]);
				
	    lm.train();	    
	    lm.readValidData();
	    lm.readTestData();

	    
//	    lm.writeMinThresholdToFile( "/Users/busarobi/work/XMLC/Posteriors/WikiLSHTC/min_threshold.txt" );
	    
	    //lm.compositeEvaluation();
	    
	    lm.outputLabels();
	    
	    lm.outputPosteriors();
	    
	}
	
	
}
