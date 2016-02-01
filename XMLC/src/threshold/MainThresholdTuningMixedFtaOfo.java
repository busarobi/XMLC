package threshold;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import IO.Evaluator;

public class MainThresholdTuningMixedFtaOfo extends MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuningMixedFtaOfo.class);
	public MainThresholdTuningMixedFtaOfo(String fname) throws IOException {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	private double[] FTAthresholds = null;
	private double[] FTAFscores = null;
	
	private double[] OFOthresholds = null;
	private double[] OFOFscores = null;

	
	public void tuneThresholdFTA() {
		this.FTAFscores = new double[this.m];
		this.FTAthresholds = new double[this.m];
		
		for( int i = 0; i < this.m; i++ ) {
			FTAFscores[i] = 0.0;
			FTAthresholds[i] = 0.0;
		}
		
		
		for( int i=0; i < this.thresholdForEUM.length; i++ ) {
			logger.info("##########################################################################");
			logger.info("Threshold: " + this.thresholdForEUM[i]);
	
			
			// set the minThreshold
			//properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
			//ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = new double[this.m];
			for( int j = 0; j < this.m; j++ ) thresholds[j] = this.thresholdForEUM[i];
			
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			double[] perf = Evaluator.computeFscores(positiveLabelsArray, this.validlabels );
			for( int im = 0; im < this.m; im++ ) {
				if (this.FTAFscores[im] < perf[im] ) {
					this.FTAFscores[im] = perf[im];
					this.FTAthresholds[im] = this.thresholdForEUM[i];
				}
				
			}

		}
		
	}
	
	
	public void tuneThresholdOFO() {		
		this.OFOFscores = new double[this.m];
		this.OFOthresholds = new double[this.m];
		
		for( int i = 0; i < this.m; i++ ) {
			this.OFOFscores[i] = 0.0;
			this.OFOthresholds[i] = 0.0;
		}
		
		for( int i=0; i < this.barray.length; i++ ) {
			logger.info("##########################################################################");
			logger.info("Threshold: " + this.barray[i]);

			
			//
			properties.setProperty("b", Integer.toString(this.barray[i]));
			
			
			ThresholdTuning tofo = new TTOfoFast( this.m, properties );
			double[] thresholds = tofo.validate(this.validlabels, this.validposteriors);

			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			double[] perf = Evaluator.computeFscores(positiveLabelsArray, this.validlabels );
			for( int im = 0; im < this.m; im++ ) {
				if (this.OFOFscores[im] < perf[im] ) {
					this.OFOFscores[im] = perf[im];
					this.OFOthresholds[im] = 1.0 / ((double)this.barray[i]);
				}
				
			}
			
		}
	}
	
	public void mergeAndEval() {
		double[] mergedThresolds = new double[this.m];
		
		for( int im = 0; im < this.m; im++ ) {
			if (this.OFOFscores[im] < this.FTAFscores[im] ) {
				mergedThresolds[im] = this.FTAthresholds[im];
			} else {
				mergedThresolds[im] = this.OFOthresholds[im];
			}
			
		}
		logger.info("##########################################################################");
		this.resultString += "##########################################################################\n";
				
		// set the minThreshold
		//properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
		//ThresholdTuning theum = new TTEumFast( this.m, properties );
		
		// compute the positive labels
		HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, mergedThresolds );
		// compute F-measure
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

		for ( String perfName : perf.keySet() ) {
			logger.info("##### OFOFTA valid " + perfName + ": "  + fmt(perf.get(perfName)));
			this.resultString += "OFOFTA,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
		}
		
		
		// compute the positive labels
		positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, mergedThresolds );
		// compute F-measure
		perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );

		for ( String perfName : perf.keySet() ) {
			logger.info("##### OFOFTA test" + perfName + ": "  + fmt(perf.get(perfName)));
			this.resultString += "OFOFTA,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
		}			
		
	}
	
	
	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}
		
		MainThresholdTuningMixedFtaOfo th = new MainThresholdTuningMixedFtaOfo(args[0]);
		
		th.loadPosteriors();
		th.addTestDataInforToResult();
		
		th.tuneThresholdFTA();
		th.tuneThresholdOFO();
		th.mergeAndEval();
		
		
		th.outFileName = th.outFileName.replace(".txt", "_ofofta.txt");
		th.writeOutResult();


	}

}
