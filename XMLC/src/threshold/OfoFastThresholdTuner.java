package threshold;

import java.util.Arrays;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import util.Constants.ThresholdTuningDictKeys;
import util.Constants.OFO;

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

		if (thresholdTunerInitOption != null && thresholdTunerInitOption.aInit != null
				&& thresholdTunerInitOption.aInit.length > 0
				&& thresholdTunerInitOption.bInit != null && thresholdTunerInitOption.bInit.length > 0) {

			setaThresholdNumerators(thresholdTunerInitOption.aInit);
			setbThresholdDenominatorst(thresholdTunerInitOption.bInit);
			logger.info("#### a[] and b[] are initialized with predefined values");

		} else {

			aThresholdNumerators = new int[numberOfLabels];
			bThresholdDenominators = new int[numberOfLabels];

			int aSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.aSeed != null
					? thresholdTunerInitOption.aSeed : OFO.defaultaSeed;

			int bSeed = thresholdTunerInitOption != null && thresholdTunerInitOption.bSeed != null
					? thresholdTunerInitOption.bSeed : OFO.defaultbSeed;

			Arrays.fill(aThresholdNumerators, aSeed);
			Arrays.fill(bThresholdDenominators, bSeed);

			logger.info("#### a[] seed: " + aSeed);
			logger.info("#### b[] seed: " + bSeed);
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
	public HashMap<Integer, Double> getTunedThresholdsSparse(Dictionary<String, Object> tuningData) throws Exception {

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
			HashSet<Integer> truePositives = trueLabels.get(j);

			for (int predictedLabel : predictedPositives) {
				bThresholdDenominators[predictedLabel]++;
				thresholdsToChange.add(predictedLabel);
			}

			for (int trueLabel : truePositives) {
				bThresholdDenominators[trueLabel]++;
				thresholdsToChange.add(trueLabel);
				if (predictedLabels.contains(trueLabel))
					aThresholdNumerators[trueLabel]++;
			}
		}

		return thresholdsToChange;
	}

	/*
	 * For single instance, this perform worse than tuneAndGetAffectedLabels,
	 * how ever, for many instances, this performs better.
	 * 
	 * For single instance: 0.87 ms by tuneAndGetAffectedLabels1, and 0.34 ms by
	 * tuneAndGetAffectedLabels (averaged over 1.0E8 iterations of random
	 * instantiations)
	 * 
	 * For 10000 instances: 1.25 ms by tuneAndGetAffectedLabels1, and 73.17 ms
	 * by tuneAndGetAffectedLabels (averaged over 1000 iterations of random
	 * instantiations)
	 * 
	 * For 10 instances: 0.9 ms by tuneAndGetAffectedLabels1, and 0.37 ms by
	 * tuneAndGetAffectedLabels (averaged over 1000000 iterations of random
	 * instantiations)
	 */
	@SuppressWarnings("unused")
	private HashSet<Integer> tuneAndGetAffectedLabels1(List<HashSet<Integer>> predictedLabels,
			List<HashSet<Integer>> trueLabels) {

		HashSet<Integer> thresholdsToChange = new HashSet<Integer>();

		for (int j = 0; j < predictedLabels.size(); j++) {

			HashSet<Integer> predictedPositives = predictedLabels.get(j);
			HashSet<Integer> truePositives = trueLabels.get(j);

			HashSet<Integer> intersection = new HashSet<Integer>(truePositives);
			intersection.retainAll(predictedPositives);

			HashSet<Integer> union = new HashSet<Integer>(truePositives);
			union.addAll(predictedPositives);

			for (int label : intersection) {
				aThresholdNumerators[label]++;
				bThresholdDenominators[label]++;
				thresholdsToChange.add(label);
			}

			for (int label : union) {
				bThresholdDenominators[label]++;
				thresholdsToChange.add(label);
			}
		}

		return thresholdsToChange;
	}
}