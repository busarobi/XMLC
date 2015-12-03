package Learner;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import Learner.step.AdamStep;
import Learner.step.GradStep;
import Learner.step.GradStepL1;
import Learner.step.StepFunction;
import preprocessing.FeatureHasher;
import preprocessing.MurmurHasher;
import threshold.TTEum;
import threshold.TTExu;
import threshold.TTOfo;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class LearnerManager {
	protected Properties properties = null;
	protected AVTable testdata =null;
	protected AVTable traindata =null;
	protected AVTable validdata =null;

	// feature hasher
	protected FeatureHasher fh = null;
	protected int featureNum = 0;

	protected AbstractLearner learner = null;

	public LearnerManager( String fname ){
		properties = readProperty(fname);

		featureNum = Integer.parseInt(properties.getProperty("FeatureHashing", "0"));

		if (featureNum>0){
			System.out.print( "Feature hashing (dim: " + featureNum + ")...");
			fh = new MurmurHasher(0, featureNum);
			System.out.println( "Done.");
		}
	}


	public Properties readProperty(String fname) {
		System.out.print("Reading property file...");
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
		System.out.println("Done.");

		return properties;
	}

	public void readTrainData() throws Exception {
		// reading train data
		DataReader datareader = new DataReader(properties.getProperty("TrainFile"));
		traindata = datareader.read();
		if (fh != null ) {
			traindata = fh.transformSparse(traindata);
		}
	}

	public void readTestData() throws Exception {
		// test
		DataReader testdatareader = new DataReader(properties.getProperty("TestFile"));
		testdata = testdatareader.read();
		if (fh != null ) {
			testdata = fh.transformSparse(testdata);
		}
	}


	public void readValidData() throws Exception {
		String validFileName = properties.getProperty("ValidFile");

		if (validFileName == null ) {
			validdata = traindata;
		} else {
			DataReader validdatareader = new DataReader(properties.getProperty("ValidFile"));
			validdata = validdatareader.read();
			if (fh != null ) {
				validdata = fh.transformSparse(validdata);
			}
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
			learner.allocateClassifiers(traindata);
			learner.train(traindata);

			String modelFile = properties.getProperty("ModelFile", null );
			if (modelFile != null ) {
				learner.savemodel(modelFile);
			}
		} else {
			learner.loadmodel(inputmodelFile);
		}

	}



	public void compositeEvaluation()
	{
		Map<String,Double> perfv = Evaluator.computePerformanceMetrics(learner, validdata);
		for ( String perfName : perfv.keySet() ) {
			System.out.println("##### EUM" + perfName + ": "  + perfv.get(perfName));
		}
		
		
		// evaluate (EUM)
		//ThresholdTuning th = new TTEum( learner.m, properties );
		//learner.tuneThreshold(th, validdata);
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);

//		// evaluate (OFO)
//		th = new TTOfo( learner.m, properties );
//		learner.tuneThreshold(th, validdata);
//		Map<String,Double> perfOFO = Evaluator.computePerformanceMetrics(learner, testdata);
//
//		// evaluate (EXU)
//		th = new TTExu( learner.m, properties );
//		learner.tuneThreshold(th, validdata);
//		Map<String,Double> perfEXU = Evaluator.computePerformanceMetrics(learner, testdata);


		for ( String perfName : perf.keySet() ) {
			System.out.println("##### EUM" + perfName + ": "  + perf.get(perfName));
		}


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
		System.out.println("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			System.err.println("No config file given!");
			System.exit(-1);
		}

		LearnerManager lm = new LearnerManager(args[0]);
		lm.readTrainData();
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
