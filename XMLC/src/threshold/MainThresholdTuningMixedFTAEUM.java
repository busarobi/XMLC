package threshold;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import IO.Evaluator;

public class MainThresholdTuningMixedFTAEUM extends MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuningMixedFTAEUM.class);
	
	public MainThresholdTuningMixedFTAEUM(String fname) throws IOException {
		super(fname);		
	}
	
	
	public void tuneThresholdFTAEUM() {		
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
			
			
			// set the minThreshold
			properties.setProperty("minThreshold", "0.0001");
			ThresholdTuning theum = new TTEumFtaFast( this.m, properties, this.thresholdForEUM);
			double[] thresholds = theum.validate(this.validlabels, this.validposteriors);
			
			this.resultString += "EUM,valid F-measure," + theum.getValidatedFmeasure() + "\n";
			
			this.resultString += "EUM,valid num. of predicted positives," + theum.getNumberOfPredictedPositives() + "\n";
			this.resultString += "EUM,valid avg. num. of predicted positives," + (theum.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";

			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EUM valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EUM,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}

			
			// compute the positive labels
			positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
			// compute F-measure
			perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### EUM " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "EUM,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			

		
		
	}

	
	
	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}
		
		MainThresholdTuningMixedFTAEUM th = new MainThresholdTuningMixedFTAEUM(args[0]);
		
		th.loadPosteriors();
		th.addTestDataInforToResult();
		
		th.tuneThresholdFTAEUM();
		
		th.outFileName = th.outFileName.replace(".txt", "_eumfta.txt");
		th.writeOutResult();
	}

}
