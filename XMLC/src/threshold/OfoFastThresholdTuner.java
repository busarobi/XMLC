package threshold;

import java.util.Arrays;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import util.Constants.ThresholdTuningDictKeys;

/**
 * Tunes thresholds using online F-Measure optimization algorithm.
 * 
 * @author Sayan
 *
 */
public class OfoFastThresholdTuner extends ThresholdTuner {
	private static Logger logger = LoggerFactory.getLogger(OfoFastThresholdTuner.class);

	protected int[] aThresholdNumerators = null;
	protected int[] bThresholdDenominators = null;

	public OfoFastThresholdTuner(int numberOfLabels, ThresholdTunerInitOption thresholdTunerInitOption) {
		super(numberOfLabels, thresholdTunerInitOption);

		logger.info("#####################################################");
		logger.info("#### OFO Fast");
		logger.info("#### numberOfLabels: " + numberOfLabels);

		if (thresholdTunerInitOption.aInit != null && thresholdTunerInitOption.aInit.length > 0
				&& thresholdTunerInitOption.bInit != null && thresholdTunerInitOption.bInit.length > 0) {

			setaThresholdNumerators(thresholdTunerInitOption.aInit);
			setbThresholdDenominatorst(thresholdTunerInitOption.bInit);
			logger.info("#### a[] and b[] are initialized with predefined values");

		} else if (thresholdTunerInitOption.aSeed != null && thresholdTunerInitOption.bSeed != null) {

			aThresholdNumerators = new int[numberOfLabels];
			bThresholdDenominators = new int[numberOfLabels];

			Arrays.fill(aThresholdNumerators, thresholdTunerInitOption.aSeed);
			Arrays.fill(bThresholdDenominators, thresholdTunerInitOption.bSeed);

			logger.info("#### a[] seed: " + thresholdTunerInitOption.aSeed);
			logger.info("#### b[] seed: " + thresholdTunerInitOption.bSeed);
		}

		logger.info("#####################################################");
	}

	private void setaThresholdNumerators(int[] aInit) {
		if (aInit.length != numberOfLabels) {
			System.exit(-1);
		}
		aThresholdNumerators = aInit;
	}

	private void setbThresholdDenominatorst(int[] bInit) {
		if (bInit.length != numberOfLabels) {
			System.exit(-1);
		}
		bThresholdDenominators = bInit;
	}

	@Override
	public double[] getTunedThresholds(Dictionary<String, Object> tuningData) {

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDictKeys.predictedLabels);

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDictKeys.trueLabels);

		if (predictedLabels != null || trueLabels != null) {

			tuneAndGetAffectedLabels(predictedLabels, trueLabels);
		}

		double[] thresholds = new double[aThresholdNumerators.length];

		for (int label = 0; label < aThresholdNumerators.length; label++) {
			thresholds[label] = (double) aThresholdNumerators[label] / (double) bThresholdDenominators[label];
		}

		return thresholds;
	}

	@Override
	HashMap<Integer, Double> getTunedThresholdsSparse(Dictionary<String, Object> tuningData) throws Exception {

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> predictedLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDictKeys.predictedLabels);

		@SuppressWarnings("unchecked")
		List<HashSet<Integer>> trueLabels = (List<HashSet<Integer>>) tuningData
				.get(ThresholdTuningDictKeys.trueLabels);

		if (predictedLabels == null || trueLabels == null)
			throw new Exception("Missing true or predicted labels");

		HashSet<Integer> thresholdsToChange = tuneAndGetAffectedLabels(predictedLabels, trueLabels);
		HashMap<Integer, Double> sparseThresholds = new HashMap<Integer, Double>();

		for (int label : thresholdsToChange) {
			sparseThresholds.put(label, (double) aThresholdNumerators[label] / (double) bThresholdDenominators[label]);
		}

		return sparseThresholds;
	}

	/**
	 * Tunes and returns the set of labels for which thresholds need to be
	 * changed.
	 * 
	 * @param predictedLabels
	 * @param trueLabels
	 * @return Set of labels for which thresholds need to be changed.
	 */
	private HashSet<Integer> tuneAndGetAffectedLabels(List<HashSet<Integer>> predictedLabels,
			List<HashSet<Integer>> trueLabels) {
		HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

		for (int j = 0; j < predictedLabels.size(); j++) {

			HashSet<Integer> predictedPositives = predictedLabels.get(j);
			HashSet<Integer> realPositives = trueLabels.get(j);

			for (int predictedLabel : predictedPositives) {
				bThresholdDenominators[predictedLabel]++;
				thresholdsToChange.add(predictedLabel);
			}

			for (int trueLabel : realPositives) {
				bThresholdDenominators[trueLabel]++;
				thresholdsToChange.add(trueLabel);
				if (predictedLabels.contains(trueLabel))
					aThresholdNumerators[trueLabel]++;
			}
		}

		return thresholdsToChange;
	}
}