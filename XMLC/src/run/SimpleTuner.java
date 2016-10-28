package run;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import IO.ReadProperty;

public class SimpleTuner {
	private static Logger logger = LoggerFactory.getLogger(SimpleTuner.class);
	// protected String resultString = "";
	protected String outFileName = "";
	protected String thresholdFileName = "";
	protected String method = "";

	protected String posteriorFileValid = null;
	protected String lableFileValid = null;

	protected AVTable validlabels = null;
	protected AVTable validposteriors = null;

	protected double[] thresholdForEUM = null;
	// protected double[] thresholdForEUM = {0.0001, 0.001, 0.005,
	// 0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5};
	protected int[] barray = { 10000, 1000, 200, 100, 50, 20, 10, 7, 5, 4, 3, 2 };

	protected int m = 0;
	// private double threshold = 0.01;
	protected Properties properties = null;
	protected boolean fastxml = false;

	public SimpleTuner(String fname) throws IOException {
		this.properties = ReadProperty.readProperty(fname);

		logger.info("#####################################################");
		logger.info("#### LearnerOutputPosteriors");

		if (!this.properties.containsKey("ValidPostFile")) {
			logger.info("No valid posterior file is given!");
			System.exit(-1);
		}

		this.posteriorFileValid = this.properties.getProperty("ValidPostFile");
		logger.info("### Valid posterior file: " + this.posteriorFileValid);

		if (!this.properties.containsKey("ValidLabelFile")) {
			logger.info("No valid posterior file is given!");
			System.exit(-1);
		}

		this.lableFileValid = this.properties.getProperty("ValidLabelFile");
		logger.info("### Valid label file: " + this.lableFileValid);

		if (!this.properties.containsKey("OutFile")) {
			logger.info("OutFile file is not given!");
			System.exit(-1);
		}

		this.outFileName = this.properties.getProperty("OutFile");
		logger.info("### OutFile: " + this.outFileName);

		if (!this.properties.containsKey("ThresholdFile")) {
			logger.info("OutFile file is not given!");
			System.exit(-1);
		}

		this.thresholdFileName = this.properties.getProperty("ThresholdFile");
		logger.info("### OutFile: " + this.thresholdFileName);

		this.method = this.properties.getProperty("Method");
		logger.info("### Method: " + this.method);

		this.fastxml = Boolean.parseBoolean(this.properties.getProperty("IsFastXML", "false"));

		String trainFile = properties.getProperty("ValidFile");
		BufferedReader brTrain = new BufferedReader(new FileReader(trainFile));
		String line = brTrain.readLine();
		brTrain.close();

		String[] tokens = line.split(" ");
		this.m = Integer.parseInt(tokens[2]);

		logger.info("#### Num. of labels: " + this.m);

		logger.info("#####################################################");

		this.thresholdForEUM = new double[this.barray.length];
		for (int i = 0; i < this.barray.length; i++) {
			this.thresholdForEUM[i] = 1.0 / ((double) this.barray[i]);
		}

	}

	protected void loadPosteriors() throws Exception {
		DataReader validddatareader = new DataReader(this.lableFileValid, false, false);
		this.validlabels = validddatareader.read();
		this.validlabels.m = this.m;

		DataReader validpostreader = new DataReader(this.posteriorFileValid, false, false);
		this.validposteriors = validpostreader.read();
		this.validposteriors.m = this.m;

		if (this.fastxml) {
			for (int i = 0; i < this.validposteriors.n; i++) {
				for (int j = 0; j < this.validposteriors.x[i].length; j++)
					this.validposteriors.x[i][j].index++;
			}
		}
	}

	public double[] tuneThresholdFTA() {

		double optimalThreshold = 0.0;
		double optimalScore = -1.0;

		for (int i = 0; i < this.thresholdForEUM.length; i++) {
			logger.info("##########################################################################");
			logger.info("Threshold: " + this.thresholdForEUM[i]);

			// set the minThreshold
			// properties.setProperty("minThreshold",
			// Double.toString(this.thresholdForEUM[i]));
			// ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = new double[this.m];
			for (int j = 0; j < this.m; j++)
				thresholds[j] = this.thresholdForEUM[i];

			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors,
					thresholds);
			// compute F-measure
			Map<String, Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels);

			for (String perfName : perf.keySet()) {
				logger.info("##### FTA valid " + perfName + ": " + fmt(perf.get(perfName)));
			}
			double score = perf.get(" macro F-measure");
			if (optimalScore < score) {
				optimalThreshold = this.thresholdForEUM[i];
				score = optimalScore;
			}
		}

		double[] retval = new double[this.m];
		for (int i = 0; i < this.m; i++)
			retval[i] = optimalThreshold;

		return retval;
	}

	public String fmt(double d) {
		if (d == (long) d)
			return String.format("%d", (long) d);
		else
			return String.format("%g", d);
	}

	public void tune() throws Exception {
		this.loadPosteriors();

		double[] thresholds = null;
		if (this.method.compareTo("FTA") == 0) {
			thresholds = this.tuneThresholdFTA();
		} else if (this.method.compareTo("STO") == 0) {
		} else if (this.method.compareTo("STO") == 0) {
		} else {
			logger.info("Unknown threshold tuning method.");
			System.exit(-1);
		}

	}

	protected HashSet<Integer>[] getPositiveLabels(AVTable labels, AVTable posteriors, double[] thresholds) {
		HashSet<Integer>[] positiveLabelsArray = new HashSet[labels.n];
		for (int i = 0; i < labels.n; i++) {
			positiveLabelsArray[i] = new HashSet<Integer>();
			for (int j = 0; j < posteriors.x[i].length; j++) {
				int labelidx = posteriors.x[i][j].index;
				double post = posteriors.x[i][j].value;
				if (post > thresholds[labelidx]) {
					positiveLabelsArray[i].add(labelidx);
				}
			}
		}
		return positiveLabelsArray;
	}


}
