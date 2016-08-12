package run;

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
import Learner.AbstractLearner;
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
	protected AVTable testdata = null;
	protected AVTable traindata = null;
	protected AVTable validdata = null;
	protected boolean isHeader = false;

	protected AbstractLearner learner = null;

	public LearnerManager(String fname) {
		properties = readProperty(fname);

		this.isHeader = Boolean.parseBoolean(properties.getProperty("IsHeader"));
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
		DataReader datareader = new DataReader(properties.getProperty("TrainFile"), false, this.isHeader);
		traindata = datareader.read();
	}

	public void readTestData() throws Exception {
		DataReader testdatareader = new DataReader(properties.getProperty("TestFile"), false, this.isHeader);
		testdata = testdatareader.read();
	}

	public void readValidData() throws Exception {
		String validFileName = properties.getProperty("ValidFile");

		if (validFileName == null) {
			validdata = traindata;
		} else {
			DataReader validdatareader = new DataReader(properties.getProperty("ValidFile"), false, this.isHeader);
			validdata = validdatareader.read();
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
		if (inputmodelFile == null) {
			this.readTrainData();
			learner.allocateClassifiers(traindata);
			learner.train(traindata);

			String modelFile = properties.getProperty("ModelFile", null);
			if (modelFile != null) {
				learner.savemodel(modelFile);
			}
		} else {
			logger.info("Loading model file...");
			this.learner = AbstractLearner.loadmodel(inputmodelFile);
		}

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

	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		LearnerManager lm = new LearnerManager(args[0]);
		// lm.readTrainData();
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
