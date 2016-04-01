package Learner;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Scanner;

import org.junit.rules.Stopwatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.ComparablePair;

public class LearnerOutputPosteriorsMeasureTimeOnTest extends LearnerOutputPosteriors {
	private static Logger logger = LoggerFactory.getLogger(LearnerOutputPosteriorsMeasureTimeOnTest.class);
	protected double[][] validatedThresholds = null;
	String[] methods = {"FTA","EUM","OFO"};
	
	public LearnerOutputPosteriorsMeasureTimeOnTest(String fname) {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	
	protected void readThresholds() throws IOException {
		String inputmodelFile = properties.getProperty("InputModelFile");
		String fname = inputmodelFile.replaceAll("PLTFHRKary.model" , "outfile_nm5_thresholds.txt");
		
		System.out.println("Reading: " + fname );

		
		Scanner input = new Scanner (new File(fname));
		// pre-read in the number of rows/columns
		int rows = 0;
		int columns = 0;
		while(input.hasNextLine())
		{
		    ++rows;
		    Scanner colReader = new Scanner(input.nextLine());
		}
		columns = 4;
		this.validatedThresholds = new double[rows][columns];

		input.close();

		// read in the data
		BufferedReader fp = new BufferedReader(new FileReader(fname));
		
		for(int i = 0; i < rows; ++i)
		{
			String line =  fp.readLine();
			String[] tokens = line.split(",");
		    for(int j = 0; j < columns; ++j)
		    {
	        	this.validatedThresholds[i][j] = Double.parseDouble(tokens[j]);
	        	//System.out.print(this.validatedThresholds[i][j] + " ");
		    }
		    //System.out.println();
		}		
		logger.info("Num x Dim " + this.validatedThresholds.length + "x" + this.validatedThresholds[0].length );
	}
	
	public void outputPosteriors() throws Exception {		
		
		//this.readTrainData();
		
		
		for ( int mi = 0; mi < methods.length; mi++ ){
			double[] thresholds = new double[this.testdata.m];
			
			
			for(int i = 0; i < this.testdata.m; i++ ) {			
				thresholds[i] = this.validatedThresholds[i][mi+1]; 
			}
			
			this.learner.setThresholds(thresholds);
		
			
			
			logger.info("############ Method: " + methods[mi] );
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

		LearnerOutputPosteriorsMeasureTimeOnTest lm = new LearnerOutputPosteriorsMeasureTimeOnTest(args[0]);
				
		lm.readThresholds();
	    lm.train();	    	    
	    lm.readTestData();

	    
//	    lm.writeMinThresholdToFile( "/Users/busarobi/work/XMLC/Posteriors/WikiLSHTC/min_threshold.txt" );
	    
	    //lm.compositeEvaluation();
	    
//	    lm.outputLabels();
	    
	    lm.outputPosteriors();


	}

}
