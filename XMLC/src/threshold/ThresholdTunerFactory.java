package threshold;

import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ThresholdTunerFactory {
	private static Logger logger = LoggerFactory.getLogger(ThresholdTunerFactory.class);

	public static ThresholdTuner createThresholdTuner(int numberOfLabels, Properties properties) {

		ThresholdTuners type = (ThresholdTuners) properties.get("tunerType");
		ThresholdTunerInitOption initOption = (ThresholdTunerInitOption) properties.get("tunerInitOption");
		ThresholdTuner retVal = null;

		switch (type) {
		case OfoFast:
			retVal = new OfoFastThresholdTuner(numberOfLabels, initOption);
			break;

		default:
			logger.info("ThresholdTuner implementation for " + type + " is not yet implmented.");
			break;
		}

		return retVal;
	}
}