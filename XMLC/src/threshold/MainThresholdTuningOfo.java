package threshold;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import IO.Evaluator;

public class MainThresholdTuningOfo extends MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuningOfo.class);
	
	private int[] aarray = {10000,1000,200,
			100,50,20,10,7,5,4,3,2,1};

	public void tuneThresholdOFO() {		
		for( int ia=0; ia < this.aarray.length; ia++ ) { 
			for( int i=1; i < this.barray.length; i++ ) {
				if ((2*this.aarray[ia]) > this.barray[i]) continue;
				
				logger.info("##########################################################################");
				this.resultString += "##########################################################################\n";
			
				logger.info("Threshold_a: " + this.aarray[ia]);
				logger.info("Threshold_b: " + this.barray[i]);
				this.resultString += "OFO,threshold_a,"+this.aarray[ia] + "\n";
				this.resultString += "OFO,threshold_b,"+this.barray[i] + "\n";
				
				//
				properties.setProperty("a", Integer.toString(this.aarray[ia]));
				properties.setProperty("b", Integer.toString(this.barray[i]));
				
				
				ThresholdTuning tofo = new TTOfoFast( this.m, properties );
				double[] thresholds = tofo.validate(this.validlabels, this.validposteriors);
				
				this.resultString += "OFO,valid F-measure," + tofo.getValidatedFmeasure() + "\n";
				
				this.resultString += "OFO,valid num. of predicted positives," + tofo.getNumberOfPredictedPositives() + "\n";
				this.resultString += "OFO,valid avg. num. of predicted positives," + (tofo.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";
		
				// compute the positive labels
				HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
				// compute F-measure
				Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );
	
				for ( String perfName : perf.keySet() ) {
					logger.info("##### OFO valid " + perfName + ": "  + fmt(perf.get(perfName)));
					this.resultString += "OFO,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
				}
				
				
				// compute the positive labels
				positiveLabelsArray = getPositiveLabels(this.testlabels, this.testposteriors, thresholds );
				// compute F-measure
				perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.testlabels );
	
				for ( String perfName : perf.keySet() ) {
					logger.info("##### OFO test" + perfName + ": "  + fmt(perf.get(perfName)));
					this.resultString += "OFO,test " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
				}			
			}
		}
	}
	
	
	
	public MainThresholdTuningOfo(String fname) throws IOException {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));

		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}

		MainThresholdTuningOfo th = new MainThresholdTuningOfo(args[0]);

		th.loadPosteriors();
		th.addTestDataInforToResult();
		
		th.tuneThresholdOFO();
		
		th.outFileName = th.outFileName.replace(".txt", "_ofo.txt");		
		th.writeOutResult();

	}

}
