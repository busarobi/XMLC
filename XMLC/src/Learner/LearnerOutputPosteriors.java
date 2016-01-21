package Learner;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.EstimatePair;
import IO.DataReader;
import IO.Evaluator;
import threshold.TTEumFast;
import threshold.TTExuFast;
import threshold.TTOfoFast;
import threshold.ThresholdTuning;

public class LearnerOutputPosteriors extends LearnerManager {
	private static Logger logger = LoggerFactory.getLogger(LearnerOutputPosteriors.class);

	private String posteriorFileValid = null;
	private String posteriorFileTest = null;
	private String lableFileValid = null;
	private String labelFileTest = null;
	
	private double threshold = 0.01;
	
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
	
	public void outputPosteriors() throws IOException {
		this.writePosteriorsToFile(this.learner, this.validdata, this.posteriorFileValid, this.threshold);
		this.writePosteriorsToFile(this.learner, this.testdata, this.posteriorFileTest, this.threshold);
	}
	
	public void writePosteriorsToFile( AbstractLearner learner, AVTable data, String fname, double minThreshold ) throws IOException{
		logger.info("Output labels to " + fname );
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
		int numOfPositives = 0;
		for( int i = 0; i<data.n; i++)
		{
			HashSet<EstimatePair> sPE = learner.getSparseProbabilityEstimates(data.x[i], minThreshold);
			
			numOfPositives += sPE.size();
			
			for(EstimatePair pred : sPE) {
				bf.write(  "" + pred.getLabel() + ":" + pred.getP() + " "  );
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
	
	public void writeLabels() throws IOException {
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
		lm.readTrainData();
	    lm.train();

	    lm.readValidData();
	    lm.readTestData();

	    lm.compositeEvaluation();
	}
	
	
}
