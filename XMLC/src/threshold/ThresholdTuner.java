package threshold;

import java.util.Map;

import util.Constants;

/**
 * Provides an interface to the Tuners. Tuners can be injected to learners,
 * where threshold tuning is an integral part of training.
 * 
 * @author Sayan
 *
 */
public abstract class ThresholdTuner {
	protected final int numberOfLabels;

	public ThresholdTuner(int numberOfLabels, ThresholdTunerInitOption thresholdTunerInitOption) {
		this.numberOfLabels = numberOfLabels;
	}

	/**
	 * Tunes thresholds as per the {@code tuningData}
	 * 
	 * @param tuningData
	 *            Data required to tune thresholds such as realLabels,
	 *            predictedLabels, probability estimates etc. Keys in the
	 *            dictionary ideally should be from
	 *            {@link Constants.ThresholdTuningDictKeys}.
	 * @return Tuned threshold array.
	 */
	abstract double[] getTunedThresholds(Map<String, Object> tuningData);

	/**
	 * Tunes thresholds as per the {@code tuningData}
	 * 
	 * @param tuningData
	 *            Data required to tune thresholds such as realLabels,
	 *            predictedLabels, probability estimates etc. Keys in the
	 *            dictionary ideally should be from
	 *            {@link Constants.ThresholdTuningDictKeys}.
	 * @return Sparse tuned thresholds, the key is label, and the value is new
	 *         value for threshold.
	 * @throws Exception
	 */
	abstract Map<Integer, Double> getTunedThresholdsSparse(Map<String, Object> tuningData)
			throws Exception;
}