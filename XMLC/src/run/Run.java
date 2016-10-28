package run;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Run {
	private static Logger logger = LoggerFactory.getLogger(Run.class);
	
	
	public static void main(String[] args) throws Exception {
		// read properties
		if (args.length < 2) {
			logger.info("The name of use case and config file should be given!");
			System.exit(-1);
		}
		
		String method = args[0];
		String configFile = args[1];
		
		logger.info( "--> Use case: " + method );
		logger.info( "--> Config file" + configFile );
		
		
		if ( method.equals("-train") ) {
			LearnerManager lm = new LearnerManager(configFile);
			lm.train();
		} else if ( method.equals("-eval") ) {
			LearnerManager lm = new LearnerManager(configFile);
			lm.eval();
		} else if ( method.equals("-posteriors") ) {
			ComputePosteriors lm = new ComputePosteriors(configFile);
			lm.computePosteriorsAndOutputLabels();
		} else if ( method.equals("-validatethresholds") ) {
			TuneThresholds th = new TuneThresholds(configFile);
			th.compositeTuning();
		} else if ( method.equals("-tune") ) {
			TuneThresholds th = new TuneThresholds(configFile);
			th.tune();							
		} else if ( method.equals("-test") ) {
			LearnerManager lm = new LearnerManager(configFile);
			lm.forecast();
		} else {
			logger.error("Unkonw method in run!!!");
			System.exit(-1);
		}
		

	}

}
