package Learner;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import preprocessing.FeatureHasher;
import threshold.TTEum;
import threshold.TTExu;
import threshold.TTOfo;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class LearnerManager {
	
	static public Properties readProperty(String fname) {
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


	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			System.err.println("No config file given!");
			System.exit(-1);			
		}
	
		Properties properties = readProperty(args[0]);
		
		// create the classifier and set the configuration
		AbstractLearner learner = null;
		String learnerName = properties.getProperty("Learner");
		
		if (learnerName.compareTo("MLLog")==0)
			learner = new MLLogisticRegression(properties);
		else if (learnerName.compareTo("MLLogNP") == 0)
			learner = new MLLogisticRegressionNSampling(properties);
		else if (learnerName.compareTo("PLT") == 0)
			learner = new PLT(properties);
		else {
			System.err.println("Unknown learner");
			System.exit(-1);
		}
			
		
		
		
		// feature hasher
		FeatureHasher fh = null;
		
		// reading train data
		DataReader datareader = new DataReader(properties.getProperty("TrainFile"));
		AVTable data = datareader.read();		

		if (properties.containsKey("FeatureHashing")) {
			int featureNum = Integer.parseInt(properties.getProperty("FeatureHashing"));
			fh = new FeatureHasher(0, featureNum);
			
			System.out.print( "Feature hashing (dim: " + featureNum + ")...");			
			data = fh.transformSparse(data);			
			System.out.println( "Done.");
		}
		
		if (properties.containsKey("seed")) {
			long seed = Long.parseLong(properties.getProperty("seed"));
			MasterSeed.setSeed(seed);
		}
		
		
		
		// train
		String inputmodelFile = properties.getProperty("InputModelFile");
		if (inputmodelFile == null ) {
			learner.allocateClassifiers(data);
			learner.train(data);

			String modelFile = properties.getProperty("ModelFile", null );
			if (modelFile != null ) {				
				learner.savemodel(modelFile);
			}
		} else {
			learner.loadmodel(inputmodelFile);
		}
		
		
		// test
		DataReader testdatareader = new DataReader(properties.getProperty("TestFile"));
		AVTable testdata = testdatareader.read();		
		if (fh != null ) {
			testdata = fh.transformSparse(testdata);			
		}
		
		String validFileName = properties.getProperty("ValidFile");
		AVTable validdata = null;
		if (validFileName == null ) {
			validdata = data;
		} else {
			DataReader validdatareader = new DataReader(properties.getProperty("ValidFile"));
			validdata = validdatareader.read();
			if (fh != null ) {
				validdata = fh.transformSparse(validdata);			
			}
			
		}
		
		
		// evaluate (EUM)
		ThresholdTuning th = new TTEum( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);
        		
		// evaluate (OFO)
		th = new TTOfo( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perfOFO = Evaluator.computePerformanceMetrics(learner, testdata);

		// evaluate (EXU)
		th = new TTExu( learner.m );
		learner.tuneThreshold(th, validdata);
		Map<String,Double> perfEXU = Evaluator.computePerformanceMetrics(learner, testdata);


		for ( String perfName : perf.keySet() ) {
			System.out.println("##### EUM" + perfName + ": "  + perf.get(perfName));
		}
		
		
		for ( String perfName : perfOFO.keySet() ) {
			System.out.println("##### OFO" + perfName + ": "  + perfOFO.get(perfName));
		}

		
		for ( String perfName : perfEXU.keySet() ) {
			System.out.println("##### EXU " + perfName + ": "  + perfEXU.get(perfName));
		}


		//learner.outputPosteriors("/Users/busarobi/work/XMLC/MLLogReg/valid_post.txt", validdata);
		//learner.outputPosteriors("/Users/busarobi/work/XMLC/MLLogReg/test_post.txt", testdata);
	}

}
