package threshold;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import IO.Evaluator;

public class MainThresholdTuningWithInit extends MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuningWithInit.class);

	public MainThresholdTuningWithInit(String fname) throws IOException {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	public void tuneThreshold( String method ) {
		logger.info("##########################################################################");
		this.resultString += "##########################################################################\n";

		
		int[] posLabels = AVTable.getNumOfLabels(this.validlabels);
		
		ThresholdTuning th = null; 
		double[] thresholds = null;
		if ( method.compareTo("OFO") == 0 ) {
			int[] binitArray = new int[this.m];
			//int[] ainitArray = new int[this.m];
			for( int i = 0; i < this.m; i++ ) {
//				binitArray[i] = posLabels[i] +  this.validposteriors.n;
				binitArray[i] = this.validposteriors.n;
			}
			// set the minThreshold
			th = new TTOfoFast(this.m, properties);
			((TTOfoFast) th).setaInit(posLabels);
			((TTOfoFast) th).setbInit(binitArray);
		
			thresholds = th.validate(this.validlabels, this.validposteriors);
			//thresholds = th.validate(this.validlabels, this.validposteriors);
		} else if ( method.compareTo("EXU") == 0 ) {
			int[] binitArray = new int[this.m];
			for( int i = 0; i < this.m; i++ ) {
//				binitArray[i] = posLabels[i] +  this.validposteriors.n;
				binitArray[i] = this.testposteriors.n;
			}
			// set the minThreshold
			th = new TTExuFast(this.m, properties);
			((TTExuFast) th).setaInit(posLabels);
			((TTExuFast) th).setbInit(binitArray);
		
			thresholds = th.validate(this.validlabels, this.validposteriors);
			//thresholds = th.validate(this.validlabels, this.validposteriors);
		}
		this.resultString += method + ",valid F-measure," + th.getValidatedFmeasure() + "\n";

		this.resultString += method + ",valid num. of predicted positives," + th.getNumberOfPredictedPositives() + "\n";
		this.resultString += method + ",valid avg. num. of predicted positives,"
				+ (th.getNumberOfPredictedPositives() / (double) this.validlabels.n) + "\n";

		// compute the positive labels
		HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds);
		// compute F-measure
		Map<String, Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels);

		for (String perfName : perf.keySet()) {
			logger.info("##### " + method +" valid " + perfName + ": " + fmt(perf.get(perfName)));
			this.resultString += method + ",valid " + perfName + "," + fmt(perf.get(perfName)) + "\n";
		}

		// compute the positive labels
		positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds);
		// compute F-measure
		perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels);

		for (String perfName : perf.keySet()) {
			logger.info("##### "+ method +" " + perfName + ": " + fmt(perf.get(perfName)));
			this.resultString += method + ",test " + perfName + "," + fmt(perf.get(perfName)) + "\n";
		}

	}

	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		MainThresholdTuningWithInit th = new MainThresholdTuningWithInit(args[0]);

		th.loadPosteriors();
		th.addTestDataInforToResult();

		th.tuneThreshold("OFO");
		th.tuneThreshold("EXU");

		th.outFileName = th.outFileName.replace(".txt", "_aut.txt");
		th.writeOutResult();
	}

}
