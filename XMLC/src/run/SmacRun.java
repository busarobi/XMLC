package run;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import Data.AVTable;
import IO.BatchDataManager;
import IO.DataManager;
import IO.DataReader;
import IO.Evaluator;
import IO.ReadProperty;
import Learner.AbstractLearner;
import util.MasterSeed;

public class SmacRun {

	private static Logger logger = LoggerFactory.getLogger(SmacRun.class);

	/*
	 * ################### SMAC Params ###################
	 */
	// // should be empty. not used, but for parsing smac stuff
	@Parameter()
	private final List<String> mainParams = new ArrayList<>();

	// just for parsing the one silly smac parameter
	@Parameter(names = "-1", hidden = true)
	private Boolean bla;

	@Parameter(names = "-gamma")
	double gamma;

	@Parameter(names = "-lambda")
	double lambda;

	@Parameter(names = "-k")
	int k;

	String trainFile;

	String testFile;

	Properties properties = new Properties();

	private DataManager traindata;

	private DataManager testdata;

	public static void main(String[] args) {
		try {
			final SmacRun main = new SmacRun();
			final JCommander jc = new JCommander(main);
			jc.parse(args);
			MasterSeed.setSeed(Long.parseLong(main.mainParams.get(4)));
			main.properties = ReadProperty.readProperty(main.mainParams.get(0));
			main.readTrainData();
			main.readTestData();
			main.run();
		} catch (Throwable e) {
			logger.error("Unexpected exception occured.",e);
			e.printStackTrace();
		}
	}

	private void run() {
		properties.setProperty("gamma", Double.toString(gamma));
		properties.setProperty("lambda", Double.toString(lambda));
		properties.setProperty("k", Integer.toString(k));
		AbstractLearner learner = AbstractLearner.learnerFactory(properties);

		learner.allocateClassifiers(traindata);
		learner.train(traindata);
//		try {
//			learner.savemodel("model.saved");
//			learner = AbstractLearner.loadmodel("model.saved");
//		} catch (IOException | ClassNotFoundException e) {
//			e.printStackTrace();
//		}
		
		Map<String, Double> perftestpreck = Evaluator.computePrecisionAtk(learner, testdata, 1);
		if (perftestpreck.size() > 1) {
			throw new IllegalArgumentException("There may be only one result for this experiment");
		} else {
			for (String perfName : perftestpreck.keySet()) {
				System.out.println("Result for SMAC: SUCCESS, 0, 0, " + (1 - perftestpreck.get(perfName)) + ", 0");
				logger.info("Performance is: {}", perftestpreck.get(perfName));
			}
		}
	}

	public void readTrainData() throws Exception {
		// reading train data
		traindata = new BatchDataManager(properties.getProperty("TrainFile"));
	}

	public void readTestData() throws Exception {
		// test
		testdata = new BatchDataManager(properties.getProperty("TestFile"));
	}

}
