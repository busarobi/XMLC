package util;

/**
 * Contains all constant values (magic strings) used in this library
 * 
 * @author Sayan
 *
 */
public class Constants {
	/**
	 * Contains constant string literals for the dictionary keys used for
	 * threshold tuning.
	 * 
	 * @author Sayan
	 *
	 */
	public static class ThresholdTuningDataKeys {
		public static final String trueLabels = "trueLabels";
		public static final String predictedLabels = "predictedLabels";
	}
	
	public static class OFO {
		public static final int defaultaSeed = 1;
		public static final int defaultbSeed = 100;
	}
}