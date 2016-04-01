package threshold;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import IO.DataReader;
import IO.Evaluator;
import Learner.AbstractLearner;

public class MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuning.class);
	protected String resultString = "";
	protected String outFileName = "";
	
	
	protected String posteriorFileValid = null;
	private String posteriorFileTest = null;
	protected String lableFileValid = null;
	private String labelFileTest = null;

	protected AVTable testlabels =null;
	protected AVTable validlabels =null;
	protected AVTable testposteriors =null;
	protected AVTable validposteriors =null;
	
	protected double[] thresholdForEUM = null;
//	protected double[] thresholdForEUM = {0.0001, 0.001, 0.005,
//										0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5};
	protected int[] barray = {10000,1000,200,
							100,50,20,10,7,5,4,3,2};
	
	protected int m = 0;
	//private double threshold = 0.01;
	protected Properties properties = null;
	protected boolean fastxml = false;
	
	
	public MainThresholdTuning(String fname) throws IOException {		
		this.properties = this.readProperty(fname);
		
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

		if (!this.properties.containsKey("OutFile")) {
			logger.info("OutFile file is not given!");
			System.exit(-1);
		}
			
		this.outFileName = this.properties.getProperty("OutFile");
		logger.info("### OutFile: " + this.outFileName );
		
		
		//this.threshold = Double.parseDouble(properties.getProperty("minThreshold", "0.001") );		
		//logger.info("#### Min threshold: " + this.threshold );
				
		this.fastxml = Boolean.parseBoolean(this.properties.getProperty("IsFastXML","false"));
		
		String trainFile = properties.getProperty("TrainFile");
		BufferedReader brTrain = new BufferedReader(new FileReader(trainFile));
	    String line = brTrain.readLine();
	    brTrain.close();
	    
	    String[] tokens = line.split(" ");
	    this.m =  Integer.parseInt(tokens[2]);
	    
	    
	    logger.info("#### Num. of labels: " + this.m );
	    
	    logger.info("#####################################################" );
	    
	    this.thresholdForEUM = new double[this.barray.length];
	    for(int i = 0; i < this.barray.length; i++ ){
	    	this.thresholdForEUM[i] = 1.0 / ((double)this.barray[i]);
	    }	    
	}
	
	
	protected void loadPosteriors() throws Exception {
		DataReader testdatareader = new DataReader(this.labelFileTest, false, false);
		this.testlabels = testdatareader.read();
		this.testlabels.m = this.m;
		
		DataReader validddatareader = new DataReader(this.lableFileValid, false, false);
		this.validlabels = validddatareader.read();
		this.validlabels.m = this.m;
		
		DataReader testpostreader = new DataReader(this.posteriorFileTest, false, false);
		this.testposteriors = testpostreader.read();
		this.testposteriors.m = this.m;
		
		if ( this.fastxml ) {
			for( int i = 0; i < this.testposteriors.n; i++ ){
				for( int j = 0; j < this.testposteriors.x[i].length; j++ )
					this.testposteriors.x[i][j].index++;
			}
		}

		logger.info("Min. post value : " + fmt(this.getMinimum(this.testposteriors)));
		
		DataReader validpostreader = new DataReader(this.posteriorFileValid, false, false);
		this.validposteriors = validpostreader.read();
		this.validposteriors.m = this.m;

		if (this.fastxml){
			for( int i = 0; i < this.validposteriors.n; i++ ){
				for( int j = 0; j < this.validposteriors.x[i].length; j++ )
					this.validposteriors.x[i][j].index++;
			}
		}
		
		logger.info("Min. valid value : " + fmt(this.getMinimum(this.validposteriors)));
		
//		logger.info("Reading vaild psoteriors");
//		this.validposteriors = readPosteriors( this.posteriorFileValid, this.validlabels.n );
//		logger.info("Reading test psoteriors");
//		this.testposteriors = readPosteriors( this.posteriorFileTest, this.testlabels.n );
	}

	protected double getMinimum(AVTable data ) {
		double min = Double.MAX_VALUE;
		
		for(int i = 0; i < data.n; i++ ){
			for( int j = 0; j < data.x[i].length; j++ ){
				if ( min > data.x[i][j].value) min = data.x[i][j].value; 
			}
		}
		
		return min;		
	}
	
	
	private HashSet<EstimatePair>[] readPosteriors( String fname, int n ) throws IOException {
		BufferedReader fp = new BufferedReader(new FileReader(fname));
		HashSet<EstimatePair>[] sPE = new HashSet[n];
		for(int i = 0; i < n; i++)
		{
			sPE[i] = new HashSet<EstimatePair>();
			
			String line = fp.readLine();
			StringTokenizer st = new StringTokenizer(line," \t\n\r\f");
			
			while (st.hasMoreTokens())
			{
				String[] tokens = st.nextToken().split(":");	
				sPE[i].add(new EstimatePair(Integer.parseInt(tokens[0]),Double.parseDouble(tokens[1]) ));
			}
		}		
		fp.close();
		
		return sPE;
	}
	
	public void tuneThresholdEUM() {		
		for( int i=0; i < this.thresholdForEUM.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
			
			logger.info("Threshold: " + this.thresholdForEUM[i]);
			this.resultString += "EUM,threshold,"+this.thresholdForEUM[i] + "\n";
			
			// set the minThreshold
			properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
			ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = theum.validate(this.validlabels, this.validposteriors);
			
			this.resultString += "EUM,valid F-measure," + theum.getValidatedFmeasure() + "\n";
			
			this.resultString += "EUM,valid num. of predicted positives," + theum.getNumberOfPredictedPositives() + "\n";
			this.resultString += "EUM,valid avg. num. of predicted positives," + (theum.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";

			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EUM valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EUM,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}

			
			// compute the positive labels
			positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EUM " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EUM,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			

		}
		
	}

	public void tuneThresholdFTA() {		
		for( int i=0; i < this.thresholdForEUM.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
			
			logger.info("Threshold: " + this.thresholdForEUM[i]);
			this.resultString += "FTA,threshold,"+this.thresholdForEUM[i] + "\n";
			
			// set the minThreshold
			//properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
			//ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = new double[this.m];
			for( int j = 0; j < this.m; j++ ) thresholds[j] = this.thresholdForEUM[i];
			
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### FTA valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "FTA,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			
			
			// compute the positive labels
			positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### FTA test" + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "FTA,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}			
			

		}
		
	}
	
	
	public void tuneThresholdOFO() {		
		
		for( int i=0; i < this.barray.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
		
			logger.info("Threshold: " + this.barray[i]);
			this.resultString += "OFO,threshold,"+this.barray[i] + "\n";
			
			//
			properties.setProperty("b", Integer.toString(this.barray[i]));
			
			
			ThresholdTuning tofo = new TTOfoFast( this.m, properties );
			double[] thresholds = tofo.validate(this.validlabels, this.validposteriors);
			
			this.resultString += "OFO,valid F-measure," + tofo.getValidatedFmeasure() + "\n";
			
			this.resultString += "OFO,valid num. of predicted positives," + tofo.getNumberOfPredictedPositives() + "\n";
			this.resultString += "OFO,valid avg. num. of predicted positives," + (tofo.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";
	
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### OFO valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "OFO,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			
			
			// compute the positive labels
			positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### OFO test" + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "OFO,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}			
		}
	}
	
	
	public void tuneThresholdEXU() {				
		
		for( int i=0; i < this.barray.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
		
			logger.info("Threshold: " + this.barray[i]);
			this.resultString += "EXU,threshold,"+this.barray[i] + "\n";
			
			//
			properties.setProperty("b", Integer.toString(this.barray[i]));
			
			
			ThresholdTuning texu = new TTExuFast( this.m, properties );
			double[] thresholds = texu.validate(this.validlabels, this.validposteriors);
			
			this.resultString += "EXU,valid F-measure," + texu.getValidatedFmeasure() + "\n";
			
			this.resultString += "EXU,valid num. of predicted positives," + texu.getNumberOfPredictedPositives() + "\n";
			this.resultString += "EXU,valid Avg. num. of predicted positives," + (texu.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";

			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EXU valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EXU,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			
			
			
			// compute the positive labels
			positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EXU test" + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EXU,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			

		}
	}
	
	public String fmt(double d)
	{
	    if(d == (long) d)
	        return String.format("%d",(long)d);
	    else
	        return String.format("%g",d);
	}

	protected HashSet<Integer>[] getPositiveLabels(AVTable labels, AVTable posteriors, double[] thresholds ){
		HashSet<Integer>[] positiveLabelsArray = new HashSet[labels.n];
		for(int i=0; i<labels.n; i++ ){
			positiveLabelsArray[i] = new HashSet<Integer>();
			for(int j=0; j < posteriors.x[i].length; j++) {
				int labelidx = posteriors.x[i][j].index;
				double post = posteriors.x[i][j].value;
				if ( post > thresholds[labelidx] ) {
					positiveLabelsArray[i].add(labelidx);
				}
			}
		}
		return positiveLabelsArray;
	}
	
	

	
	

	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}
		
		MainThresholdTuning th = new MainThresholdTuning(args[0]);
		
		th.loadPosteriors();
		th.addTestDataInforToResult();
		
		th.tuneThresholdFTA();
		th.tuneThresholdEUM();
		th.tuneThresholdOFO();
		//th.tuneThresholdEXU();
		
		th.writeOutResult();
	}

	protected void addTestDataInforToResult() {
		this.resultString += "##########################################################################\n";
		
		this.resultString += "GEN,Num of labels,"+this.m + "\n";
		
		int numOfPostiveLabel = 0;
		
		for( int i = 0; i < this.validlabels.n; i++ ){
			numOfPostiveLabel += this.validlabels.y[i].length;
		}
		
		this.resultString += "GEN,valid num of positive labels,"+ numOfPostiveLabel + "\n";
		this.resultString += "GEN,valid avg. num of positive labels,"+ (numOfPostiveLabel / (double) this.validlabels.n )+ "\n";

		
		numOfPostiveLabel = 0;
		
		for( int i = 0; i < this.testlabels.n; i++ ){
			numOfPostiveLabel += this.testlabels.y[i].length;
		}
		
		this.resultString += "GEN,test num of positive labels,"+ numOfPostiveLabel + "\n";
		this.resultString += "GEN,test avg. num of positive labels,"+ (numOfPostiveLabel / (double) this.testlabels.n )+ "\n";
		
	}
	
	
	protected void writeOutResult() throws IOException {
		BufferedWriter bf = new BufferedWriter(new FileWriter(this.outFileName) );
		bf.write(this.resultString);
		bf.close();
	}
	
	
	public Properties readProperty(String fname) {
		logger.info("Reading property file...");
		Properties properties = new Properties();
		try {
			FileInputStream in = new FileInputStream(fname);
			properties.load(in);
			in.close();
		} catch (FileNotFoundException e) {
			System.err.println(e.getMessage());
			System.exit(-1);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			System.exit(-1);
		}
		logger.info("Done.");

		return properties;
	}
	
	
}
