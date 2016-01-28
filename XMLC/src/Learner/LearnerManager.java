package Learner;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import preprocessing.FeatureHasher;
import threshold.TTEum;
import threshold.TTEumFast;
import threshold.TTExuFast;
import threshold.TTOfoFast;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class LearnerManager {
	private static Logger logger = LoggerFactory.getLogger(LearnerManager.class);
	
	protected Properties properties = null;
	protected AVTable testdata =null;
	protected AVTable traindata =null;
	protected AVTable validdata =null;
	protected boolean isHeader = false;
	
	// feature hasher
	protected FeatureHasher fh = null;
	protected int featureNum = 0;

	protected AbstractLearner learner = null;

	public LearnerManager( String fname ){
		properties = readProperty(fname);

		this.isHeader = Boolean.parseBoolean(properties.getProperty("IsHeader"));
//		featureNum = Integer.parseInt(properties.getProperty("FeatureHashing", "0"));

//		if (featureNum>0){
//			System.out.print( "Feature hashing (dim: " + featureNum + ")...");
//			fh = new MurmurHasher(0, featureNum);
//			System.out.println( "Done.");
//		}
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

	public void readTrainData() throws Exception {
		// reading train data
		DataReader datareader = new DataReader(properties.getProperty("TrainFile"), false, this.isHeader);
		traindata = datareader.read();
//		if (fh != null ) {
//			traindata = fh.transformSparse(traindata);
//		}
	}

	public void readTestData() throws Exception {
		// test
		DataReader testdatareader = new DataReader(properties.getProperty("TestFile"),false, this.isHeader );
		testdata = testdatareader.read();
//		if (fh != null ) {
//			testdata = fh.transformSparse(testdata);
//		}
	}


	public void readValidData() throws Exception {
		String validFileName = properties.getProperty("ValidFile");

		if (validFileName == null ) {
			validdata = traindata;
		} else {
			DataReader validdatareader = new DataReader(properties.getProperty("ValidFile"), false, this.isHeader);
			validdata = validdatareader.read();
//			if (fh != null ) {
//				validdata = fh.transformSparse(validdata);
//			}
		}
	}

	public void train() throws Exception {
		this.learner = AbstractLearner.learnerFactory(properties);
		

		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}

		// train
		String inputmodelFile = properties.getProperty("InputModelFile");
		if (inputmodelFile == null ) {
			this.readTrainData();
			learner.allocateClassifiers(traindata);
			learner.train(traindata);

			String modelFile = properties.getProperty("ModelFile", null );
			if (modelFile != null ) {
				learner.savemodel(modelFile);
			}
		} else {
			logger.info("Loading model file...");
			this.learner = AbstractLearner.loadmodel(inputmodelFile);
		}

	}



	public void compositeEvaluation()
	{

//		double [] thresholds = {/*0.0, 0.01,*/ 0.05, 0.1, 0.25, 0.5};
//		Map<String,Double> [] perf = new Map[thresholds.length];
//		
//		for(int t = 0; t < thresholds.length ; t++){
//
//			this.learner.setThresholds(thresholds[t]);
//			perf[t] = Evaluator.computePerformanceMetrics(learner, testdata);
//
//		}
		
		Map<String,Double> perfvalidpreck = Evaluator.computePrecisionAtk(this.learner, this.validdata, 5);
		Map<String,Double> perftestpreck = Evaluator.computePrecisionAtk(this.learner, this.testdata, 5);
		
		//Map<String,Double> perfpreck = Evaluator.computePrecisionAtk(learner, testdata, 5);
		
		for ( String perfName : perfvalidpreck.keySet() ) {
			logger.info("##### Valid " + perfName + ": "  + perfvalidpreck.get(perfName) );			
		}

		for ( String perfName : perftestpreck.keySet() ) {
			logger.info("##### Test " + perfName + ": "  + perftestpreck.get(perfName));			
		}
		
		
//		// evaluate (EUM)
//		ThresholdTuning theum = new TTEumFast( learner.m, properties );
//		learner.tuneThreshold(theum, validdata);
//		Map<String,Double> perfTTEUMFast = Evaluator.computePerformanceMetrics(learner, testdata);
//							
//		// evaluate (OFO)
//		ThresholdTuning th = new TTOfoFast( learner.m, properties );
//		learner.tuneThreshold(th, validdata);			
//		Map<String,Double> perfTTOFOFast = Evaluator.computePerformanceMetrics(learner, testdata);
//		
//		// evaluate (EXU)
//		ThresholdTuning thexu = new TTExuFast( learner.m, properties );
//		learner.tuneThreshold(thexu, validdata);
//		Map<String,Double> perfTTExu = Evaluator.computePerformanceMetrics(learner, testdata);		
//		
//		
//		for ( String perfName : perfTTEUMFast.keySet() ) {
//			logger.info("##### EUM " + perfName + ": "  + perfTTEUMFast.get(perfName));
//		}
//		
//		
//		for ( String perfName : perfTTOFOFast.keySet() ) {
//			logger.info("##### OFO " + perfName + ": "  + perfTTOFOFast.get(perfName));
//		}
//
//
//		for ( String perfName : perfTTExu.keySet() ) {
//			logger.info("##### EXU " + perfName + ": "  + perfTTExu.get(perfName));
//		}
		
		
//		for(int t = 0; t < thresholds.length; t++){
//			System.out.println("##########-----  Threshold: " + thresholds[t]);
//			for ( String perfName : perf[t].keySet() ) {
//				System.out.println("##### EUM" + perfName + ": "  + perf[t].get(perfName));
//			}
//		}
	
		
		
		
		
//		// evaluate (OFO)
//		th = new TTOfo( learner.m, properties );
//		learner.tuneThreshold(th, validdata);
//		Map<String,Double> perfOFO = Evaluator.computePerformanceMetrics(learner, testdata);
//
//		// evaluate (EXU)
//		th = new TTExu( learner.m, properties );
//		learner.tuneThreshold(th, validdata);
//		Map<String,Double> perfEXU = Evaluator.computePerformanceMetrics(learner, testdata);




//		for ( String perfName : perfOFO.keySet() ) {
//			System.out.println("##### OFO" + perfName + ": "  + perfOFO.get(perfName));
//		}
//
//
//		for ( String perfName : perfEXU.keySet() ) {
//			System.out.println("##### EXU " + perfName + ": "  + perfEXU.get(perfName));
//		}


		//learner.outputPosteriors("/Users/busarobi/work/XMLC/MLLogReg/valid_post.txt", validdata);
		//learner.outputPosteriors("/Users/busarobi/work/XMLC/MLLogReg/test_post.txt", testdata);
	}

	public Map<String,Double> test(){
		// evaluate (EUM)
		ThresholdTuning th = new TTEum( learner.m, properties );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);

		return perf;
	}


	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		LearnerManager lm = new LearnerManager(args[0]);
		//lm.readTrainData();
	    lm.train();

	    lm.readValidData();
	    lm.readTestData();

	    lm.compositeEvaluation();

	}


	public Properties getProperties() {
		return properties;
	}


	public void setProperties(Properties properties) {
		this.properties = properties;
	}


}
