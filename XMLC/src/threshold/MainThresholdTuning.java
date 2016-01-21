package threshold;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
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
	
	private String posteriorFileValid = null;
	private String posteriorFileTest = null;
	private String lableFileValid = null;
	private String labelFileTest = null;

	protected AVTable testlabels =null;
	protected AVTable validlabels =null;
	protected AVTable testposteriors =null;
	protected AVTable validposteriors =null;
	
	private double[] thresholdForEUM = {0.01,0.05,0.1,0.15,0.2};
	private int m = 0;
	private double threshold = 0.01;
	protected Properties properties = null;
	
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

		this.threshold = Double.parseDouble(properties.getProperty("minThreshold", "0.001") );
		
		logger.info("#### Min threshold: " + this.threshold );
				
		
		
		String trainFile = properties.getProperty("TrainFile");
		BufferedReader brTrain = new BufferedReader(new FileReader(trainFile));
	    String line = brTrain.readLine();
	    brTrain.close();
	    
	    String[] tokens = line.split(" ");
	    this.m =  Integer.parseInt(tokens[2]);
	    
	    
	    logger.info("#### Num. of labels: " + this.m );
	    
	    logger.info("#####################################################" );
	}
	
	private void loadPosteriors() throws Exception {
		DataReader testdatareader = new DataReader(this.labelFileTest, false, false);
		this.testlabels = testdatareader.read();
		
		DataReader validddatareader = new DataReader(this.lableFileValid, false, false);
		this.validlabels = validddatareader.read();

		DataReader testpostreader = new DataReader(this.posteriorFileTest, false, false);
		this.testposteriors = testpostreader.read();

		DataReader validpostreader = new DataReader(this.posteriorFileValid, false, false);
		this.validposteriors = validpostreader.read();
		
		
		
//		logger.info("Reading vaild psoteriors");
//		this.validposteriors = readPosteriors( this.posteriorFileValid, this.validlabels.n );
//		logger.info("Reading test psoteriors");
//		this.testposteriors = readPosteriors( this.posteriorFileTest, this.testlabels.n );
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
	
	public double[] tuneThresholdEUM() {		
		double[] scores = new double[this.thresholdForEUM.length];
		
		for( int i=0; i < this.thresholdForEUM.length; i++ ) {
			logger.info("##########################################################################");
			logger.info("Threshold: " + this.thresholdForEUM[i]);
			properties.setProperty("minThreshold", Double.toHexString(this.thresholdForEUM[i]));
			ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = theum.validate(this.validlabels, this.validposteriors);
			
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = this.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### " + perfName + ": "  + perf.get(perfName));
			}

		}
		
		return scores;
	}

	protected HashSet<Integer>[] getPositiveLabels(AVTable labels, AVTable posteriors, double[] thresholds ){
		HashSet<Integer>[] positiveLabelsArray = new HashSet[labels.n];
		for(int i=0; i<testlabels.n; i++ ){
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
	
	
	
    public Map<String,Double> computePerformanceMetrics(HashSet<Integer>[] positiveLabelsArray, AVTable data) {
		logger.info("--> Computing Hamming loss and F-measure...");

		double macroF = 0.0;
		
		
		int[] tp = new int[this.m];
		int[] yloc = new int[this.m];
		int[] haty = new int[this.m];
		
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
					if (trueLabel < this.m ) { // this label was seen in the training
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
					if (trueLabel>= this.m) continue; // this label is not seen in the training
					
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
		double normalizedHL = (HL / (double)this.m);
		
		
		int presentedlabels = 0;
		for(int i = 0; i < this.m; i++) {
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
		arr.put( " learner.m", (double) this.m);
		arr.put( " Num of presented labels", (double) presentedlabels);
		

		arr.put(" Normalized macro F-measue (with presented labels)", normalizedmacroF);
		arr.put(" Normalized Hamming loss (with learner.m)", normalizedHL );

		
		return arr;

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
		
		th.tuneThresholdEUM();
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
