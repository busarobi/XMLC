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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import IO.Evaluator;
import util.MasterSeed;

public class TuneHyperParameters extends LearnerManager {
	private static Logger logger = LoggerFactory.getLogger(TuneHyperParameters.class);

	protected HashMap<String, List<String> > hyperparameters = new HashMap<String, List<String> >();
	protected Random rand = new Random();
	protected int numWorkers = 8;
	protected int numOfTrials = 100;


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
			this.learner = AbstractLearner.learnerFactory(properties);
			
			if (properties.containsKey("seed")) {
				long seed = Long.parseLong(properties.getProperty("seed"));
				MasterSeed.setSeed(seed);
			}

			// train
			learner.allocateClassifiers(traindata);
			learner.train(traindata);
			
			
			Map<String,Double> perfvalidpreck = Evaluator.computePrecisionAtk(learner, validdata, 5);
			Map<String,Double> perftestpreck = Evaluator.computePrecisionAtk(learner, testdata, 5);
			
			// valid			
//			ThresholdTuning th = new TTEumFast( learner.m, properties );
//			learner.tuneThreshold(th, validdata);
//
//			Map<String,Double> perfv = Evaluator.computePerformanceMetrics(learner, validdata);
//
			
			// generate result
			this.info += "#### Valid:\n";
			
//			for ( String perfName : perfv.keySet() ) {
//				logger.info("##### Valid " + perfName + ": "  + perfv.get(perfName));
//				this.info += "##### Valid" + perfName + ": "  + perfv.get(perfName) + "\n";
//			}
//			
//			
			for ( String perfName : perfvalidpreck.keySet() ) {
				logger.info("##### Valid " + perfName + ": "  + perfvalidpreck.get(perfName) );
				this.info += "##### Valid " + perfName + ": "  + perfvalidpreck.get(perfName) + "\n";
			}
			
			
			this.info += "#### Test:\n";
			
//			Map<String,Double> perf = Evaluator.computePerformanceMetrics(learner, testdata);
//
//			// generate result
//			for ( String perfName : perf.keySet() ) {
//				logger.info("##### Test " + perfName + ": "  + perf.get(perfName));
//				this.info += "##### Test" + perfName + ": "  + perf.get(perfName) + "\n";
//			}
			
			
			
			for ( String perfName : perftestpreck.keySet() ) {
				logger.info("##### Test " + perfName + ": "  + perftestpreck.get(perfName));
				this.info += "##### Test " + perfName + ": "  + perftestpreck.get(perfName) + "\n";
			}
			
			
			learner = null;
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

	public TuneHyperParameters(String pfname ) {
		super(pfname);
		try{

			// gamma
			List<String> gammaArray = Arrays.asList("100.0","70.0","50.0","40.0","30.0","20.0","10.0","5.0","1.0","0.5","0.1","0.05","0.01","0.005","0.001","0.0001","0.00001","0.000001");
			//List<String> gammaArray = Arrays.asList("10.0","5.0","1.0","0.5","0.1","0.05","0.01","0.005","0.001","0.0001","0.00001","0.000001");
			//List<String> gammaArray = Arrays.asList("100.0","70.0","50.0","40.0","30.0","20.0"); //"10.0","5.0","1.0","0.5","0.1","0.01","0.001");
			hyperparameters.put("gamma", gammaArray);

			// lambda
			List<String> lambdaArray = Arrays.asList("0.5","0.1","0.05","0.01","0.005","0.001","0.0001","0.00001","0.000001","0.0000001");
			//List<String> lambdaArray = Arrays.asList("0.0001","0.00001","0.000001","0.0000001");
			hyperparameters.put("lambda", lambdaArray);
			

			// number of leaves
			//List<String> kArray = Arrays.asList("2","4","8","16","32","64");
			List<String> kArray = Arrays.asList("2","3","4");
			//List<String> kArray = Arrays.asList("2","4","8");
			//List<String> lambdaArray = Arrays.asList("0.0001","0.00001","0.000001","0.0000001");
			hyperparameters.put("k", kArray);
			
			
//			// step
//			List<String> stepArray = Arrays.asList("50000","30000","20000","10000","5000","2000","1000","500","200","100","50","10");
//			hyperparameters.put("step", stepArray);
//
//			// epochs
			List<String> epochArray = Arrays.asList("10","20","30","50","100");
			//List<String> epochArray = Arrays.asList("10","50","100");
			//List<String> epochArray = Arrays.asList("5","10","15","20");
			//List<String> epochArray = Arrays.asList("1");
			//List<String> epochArray = Arrays.asList("1");
			hyperparameters.put("epochs", epochArray);			
			
			
//			List<String> beta1Array = Arrays.asList("0.7", "0.8", "0.9");
//			hyperparameters.put("beta1", beta1Array );
//
//			List<String> beta2Array = Arrays.asList("0.999", "0.9", "0.8");
//			hyperparameters.put("beta2", beta2Array );

//			List<String> etaArray = Arrays.asList("100.0","70.0","50.0","40.0","30.0","20.0","10.0","5.0","1.0","0.5","0.1","0.05","0.01","0.005","0.001","0.0001","0.00001","0.000001");			
//			hyperparameters.put("eta", etaArray);
//
//			List<String> lambda0Array = Arrays.asList("0.1","0.01","0.005","0.001","0.0001","0.00001","0.000001","0.0000001","0.00000001", "0.0");			
//			hyperparameters.put("lambda0", lambda0Array);
//			
//			List<String> lambda1Array = Arrays.asList("0.1","0.01","0.005","0.001","0.0001","0.00001","0.000001","0.0000001","0.00000001", "0.0");			
//			hyperparameters.put("lambda1", lambda1Array);
			
			logger.info("#####################################################" );
			logger.info("#### Random hyperparameter tunning " );

			// number of workers
			this.numWorkers = Integer.parseInt(this.properties.getProperty("numWorkers", "4"));
			logger.info("#### num of workers: " + this.numWorkers );

			// number of workers
			this.numOfTrials = Integer.parseInt(this.properties.getProperty("numOfTrials", "100"));
			logger.info("#### num of trials: " + this.numOfTrials );

			logger.info("#####################################################" );
			
		} catch (Exception e ){
			logger.info(e.getMessage());
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

		
		this.readTrainData();
		this.readValidData();
		this.readTestData();


		ExecutorService executor = Executors.newFixedThreadPool(this.numWorkers);//creating a pool of 5 threads
		SimpleThread[] workers = new SimpleThread[numOfTrials];

		for( int hpi = 0; hpi < numOfTrials; hpi++ ){
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
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 2) {
			System.err.println("No config and log file given!");
			System.exit(-1);
		}

		TuneHyperParameters tuner = new TuneHyperParameters(args[0]);
		tuner.run( args[1] );
	}

}
