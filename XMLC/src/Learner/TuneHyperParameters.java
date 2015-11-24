package Learner;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import Data.AVTable;
import IO.Evaluator;
import Learner.step.AdamStep;
import Learner.step.StepFunction;
import threshold.TTEum;
import threshold.ThresholdTuning;
import util.MasterSeed;

public class TuneHyperParameters extends LearnerManager {
	protected HashMap<String, List<String> > hyperparameters = new HashMap<String, List<String> >();
	protected Random rand = new Random();


	class SimpleThread implements Runnable {
		protected Properties properties = null;
		protected AVTable testdata =null;
		protected AVTable traindata =null;
		protected AVTable validdata =null;
		protected String info = "";
		protected boolean ready = false;

		public SimpleThread( Properties prop, AVTable train, AVTable valid, AVTable test, String info ){
			this.properties = prop;
			this.traindata = train;
			this.validdata = valid;
			this.testdata = test;
			this.info = info;
		}

		protected AbstractLearner learner = null;

		@Override
		public void run() {
			// Create step function:
			StepFunction stepfunction;
			String stepName = properties.getProperty("StepFunction");
			if (stepName.compareTo("Adam") == 0)
				stepfunction = new AdamStep(properties);
			else {
				stepfunction = new AdamStep(properties);
			}
			// create the classifier and set the configuration
			String learnerName = properties.getProperty("Learner");

			if (learnerName.compareTo("MLLog")==0)
				learner = new MLLogisticRegression(properties, stepfunction);
			else if (learnerName.compareTo("MLLogNP") == 0)
				learner = new MLLogisticRegressionNSampling(properties, stepfunction);
			else if (learnerName.compareTo("PLT") == 0)
				learner = new PLT(properties, stepfunction);
			else {
				System.err.println("Unknown learner");
				System.exit(-1);
			}

			if (properties.containsKey("seed")) {
				long seed = Long.parseLong(properties.getProperty("seed"));
				MasterSeed.setSeed(seed);
			}

			// train
			learner.allocateClassifiers(traindata);
			learner.train(traindata);
			// test
			ThresholdTuning th = new TTEum( learner.m, properties );
			learner.tuneThreshold(th, validdata);
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);

			// generate result
			for ( String perfName : perf.keySet() ) {
				System.out.println("##### " + perfName + ": "  + perf.get(perfName));
				this.info += "##### " + perfName + ": "  + perf.get(perfName) + "\n";
			}
			ready = true;
		}
		public String getInfo() {
			return info;
		}

		public boolean isReady() {
			return this.ready;
		}

	}

	@Override
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

	public TuneHyperParameters(String pfname ) {
		super(pfname);
		try{

			this.readTrainData();
			this.readValidData();
			this.readTestData();

			// gamma
			List<String> gammaArray = Arrays.asList("100.0","70.0","50.0","40.0","30.0","20.0","10.0","5.0","1.0","0.5","0.1","0.05","0.01","0.005","0.001");
			hyperparameters.put("gamma", gammaArray);
			// step
			List<String> stepArray = Arrays.asList("50000","30000","20000","10000","5000","2000","1000","500","200","100","50","10");
			hyperparameters.put("step", stepArray);
			// delta
			//List<String> deltaArray = Arrays.asList("0.0","0.1","0.01");
			List<String> deltaArray = Arrays.asList("0.0");
			hyperparameters.put("delta", deltaArray);

			// epochs
			List<String> epochArray = Arrays.asList("50","100","200","500");
			//List<String> epochArray = Arrays.asList("2","3","4","5");
			hyperparameters.put("epochs", epochArray);


		} catch (Exception e ){
			System.out.println(e.getMessage());
		}

	}


	public Properties getUpdatedProperties() {
		// update the properties
		Properties prop = (Properties) this.properties.clone();
		for(String featName : this.hyperparameters.keySet() ) {
			List<String> arr = this.hyperparameters.get(featName);
			int r = rand.nextInt(arr.size());
			String value = new String("" + arr.get(r));
			prop.put(featName, value);

			//result.concat( "##### " + featName + " " + value+ "\n" );
		}
		return prop;
	}

	public String getInfoString(Properties prop)  {

		String result = "";
		result += "#######################################################\n";


		for(String featName : this.hyperparameters.keySet() ) {
			String value = prop.getProperty(featName);
			result += "##### " + featName + " " + value+ "\n";
		}

		return result;
	}

	public void run (String fname) throws Exception{

		int numWorkers = 12;
		int numOfTrial = 100;

		ExecutorService executor = Executors.newFixedThreadPool(numWorkers);//creating a pool of 5 threads
		SimpleThread[] workers = new SimpleThread[numOfTrial];

		for( int hpi = 0; hpi < numOfTrial; hpi++ ){
			Properties prop = getUpdatedProperties();
			String info = getInfoString(prop);

			workers[hpi] = new SimpleThread( prop, traindata, validdata, testdata, info );
			executor.execute(workers[hpi]);
		}

		executor.shutdown();

		Writer writer = null;
		writer = new BufferedWriter(new OutputStreamWriter(
		          new FileOutputStream(fname), "utf-8"));
		// output
		writer.write("Train Filename: " + this.properties.getProperty("TrainFile") + "\n");
		writer.write("Test Filename: " + this.properties.getProperty("TestFile") + "\n");
		writer.write("Learner: " + this.properties.getProperty("Learner") + "\n");
		writer.flush();


		while (!executor.isTerminated()) {
			Thread.sleep(1000);
			for( int i = 0; i < workers.length; i++ ) {
				if (workers[i] != null ) {
					if (workers[i].isReady()) {
						writer.write( "#######################################################\n" );
						writer.write( "------------> JOB: " + (i+1) + " (" + workers.length + ")" +"\n" );
						writer.write(workers[i].getInfo());
						writer.flush();

						workers[i] = null;
					}
				}
			}

		}
//		for( int hpi = 0; hpi < numOfTrial; hpi++ ){
//			writer.write(workers[hpi].getInfo());
//			writer.flush();
//		}

		writer.close();
	}

	public static void main(String[] args) throws Exception {
		System.out.println("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 2) {
			System.err.println("No config and log file given!");
			System.exit(-1);
		}

		TuneHyperParameters tuner = new TuneHyperParameters(args[0]);
		tuner.run( args[1] );
	}

}
