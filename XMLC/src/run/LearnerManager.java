package run;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import Data.Instance;
import IO.BatchDataManager;
import IO.DataManager;
import IO.DataReader;
import IO.Evaluator;
import IO.ReadProperty;
import Learner.AbstractLearner;
import threshold.TTEum;
import threshold.TTEumFast;
import threshold.TTExuFast;
import threshold.TTOfoFast;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class LearnerManager {
	private static Logger logger = LoggerFactory.getLogger(LearnerManager.class);

	protected Properties properties = null;
	protected DataManager testdata = null;
	protected DataManager traindata = null;
	protected DataManager validdata = null;
	protected boolean isHeader = true;

	protected AbstractLearner learner = null;

	public LearnerManager(String fname) {
		properties = ReadProperty.readProperty(fname);

		this.isHeader = Boolean.parseBoolean(properties.getProperty("IsHeader"));
	}


	public void readTrainData() throws Exception {		
		//traindata = new BatchDataManager(properties.getProperty("TrainFile"));
		traindata = DataManager.managerFactory(properties.getProperty("TrainFile"), "Online" );
	}

	public void readTestData() throws Exception {		
		//testdata = new BatchDataManager(properties.getProperty("TestFile"));
		testdata = DataManager.managerFactory(properties.getProperty("TestFile"), "Online" );
	}

	public void readValidData() throws Exception {
		String validFileName = properties.getProperty("ValidFile");

		if (validFileName == null) {
			validdata = traindata;
		} else {			
			//validdata = new BatchDataManager(properties.getProperty("ValidFile"));
			validdata = DataManager.managerFactory(properties.getProperty("ValidFile"), "Online" );
		}
	}

	public void trainorrelod() throws Exception {
		this.learner = AbstractLearner.learnerFactory(properties);

		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}

		// train
		String inputmodelFile = properties.getProperty("InputModelFile");
		if (inputmodelFile == null) {
			this.readTrainData();
			learner.allocateClassifiers(traindata);
			learner.train(traindata);
			this.traindata.close();
			
			String modelFile = properties.getProperty("ModelFile", null);
			if (modelFile != null) {
				learner.savemodel(modelFile);
			}
		} else {
			logger.info("Loading model file...");
			this.learner = AbstractLearner.loadmodel(inputmodelFile);
		}

	}

	public void train() throws Exception {
		this.learner = AbstractLearner.learnerFactory(properties);

		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}

		// train
		this.readTrainData();
		learner.allocateClassifiers(traindata);
		learner.train(traindata);
		traindata.close();
		
		String modelFile = properties.getProperty("ModelFile", null);
		if (modelFile != null) {
			logger.info("Saving model file to " + modelFile );
			learner.savemodel(modelFile);			
		}

	}
	
	
	public void forecast() throws Exception {
		this.readTestData();
		this.loadmodel();
		double[] thresholds = this.readThreshold();
		this.learner.setThresholds(thresholds);
				
		String outFile = properties.getProperty("OutFile", null);
		logger.info("Print forecast to " + outFile );
		BufferedWriter bf = new BufferedWriter(new FileWriter(outFile) );
		while(this.testdata.hasNext() == true ) {		
			Instance instance = this.testdata.getNextInstance();
			HashSet<Integer> posLabels = this.learner.getPositiveLabels( instance.x );
			for( Integer lab : posLabels ){
				bf.write("" + lab + " " );
			}
			bf.write("\n");
		}
		bf.close();
		this.testdata.close();
	}
	
	
	public double[] readThreshold() throws Exception {
		String thresholdFile = properties.getProperty("ThresholdFile", null);
		double[] thresholds = new double[this.learner.getNumberOfLabels()];
		
		
		BufferedReader fp = new BufferedReader(new FileReader(thresholdFile));
		for(int i = 0; i < this.learner.getNumberOfLabels(); i++ ) {
			thresholds[i] = Double.parseDouble(fp.readLine());
		}		
		fp.close();
		
		return thresholds;
	}
	
	
	public void compositeEvaluation() {

		Map<String, Double> perfvalidpreck = Evaluator.computePrecisionAtk(this.learner, this.validdata, 5);
		Map<String, Double> perftestpreck = Evaluator.computePrecisionAtk(this.learner, this.testdata, 5);

		// Map<String,Double> perfpreck = Evaluator.computePrecisionAtk(learner,
		// testdata, 5);

		for (String perfName : perfvalidpreck.keySet()) {
			logger.info("##### Valid " + perfName + ": " + perfvalidpreck.get(perfName));
		}

		for (String perfName : perftestpreck.keySet()) {
			logger.info("##### Test " + perfName + ": " + perftestpreck.get(perfName));
		}

	}

	public void run() throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));
		
		// lm.readTrainData();
		this.trainorrelod();

		this.readValidData();
		this.readTestData();

		this.compositeEvaluation();
		
	}

	public void loadmodel() throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		this.learner = AbstractLearner.learnerFactory(properties);

		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}

		// train
		String inputmodelFile = properties.getProperty("InputModelFile");
		if (inputmodelFile == null) {
			logger.info("No model file is given!!!");
			System.exit(-1);
		} else {
			logger.info("#################################################### ");
			logger.info("Loading model file from " + inputmodelFile);
			this.learner = AbstractLearner.loadmodel(inputmodelFile);
			logger.info("Parameters of the loaded model ");
			this.learner.printParameters();
			logger.info("#################################################### ");
		}
		
	
	}	
	
	public void traineval() throws Exception{
		this.learner = AbstractLearner.learnerFactory(properties);

		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}

		// train
		this.readTrainData();
		learner.allocateClassifiers(traindata);
		learner.train(traindata);
		traindata.close();

		this.readTestData();
		Map<String, Double> perftestpreck = Evaluator.computePrecisionAtk(this.learner, this.testdata, 5);
		
		for (String perfName : perftestpreck.keySet()) {
			logger.info("##### Test " + perfName + ": " + perftestpreck.get(perfName));
		}	
		this.testdata.close();		
	}
	
	
	public void eval() throws Exception {		
		this.loadmodel();		
		this.readTestData();
		Map<String, Double> perftestpreck = Evaluator.computePrecisionAtk(this.learner, this.testdata, 5);
		
		for (String perfName : perftestpreck.keySet()) {
			logger.info("##### Test " + perfName + ": " + perftestpreck.get(perfName));
		}	
		this.testdata.close();
	}
	
	
	public static void main(String[] args) throws Exception {
		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}
		
		LearnerManager lm = new LearnerManager(args[0]);
		lm.run();
	}

	public Properties getProperties() {
		return properties;
	}

	public void setProperties(Properties properties) {
		this.properties = properties;
	}

}
