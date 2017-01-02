package threshold;

/**
 * Provides a set of optional properties that are needed to initialize an
 * instance of {@link ThresholdTuner}
 * 
 * @author Sayan
 *
 */
public class ThresholdTunerInitOption {
	/**
	 * Single seed value for aThresholdNumerators (OFO).
	 */
	public Integer aSeed;
	/**
	 * Single seed value for bThresholdDenominators (OFO).
	 */
	public Integer bSeed;
	/**
	 * Initial preset value for aThresholdNumerators (OFO).
	 */
	public int[] aInit;
	/**
	 * Initial preset value for bThresholdDenominators (OFO).
	 */
	public int[] bInit;
}
