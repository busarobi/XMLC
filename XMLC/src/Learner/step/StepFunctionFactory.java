package Learner.step;

import java.util.Properties;

public class StepFunctionFactory {
	public static StepFunction factory( Properties properties ){
		StepFunction stepfunction;
		String stepName = properties.getProperty("StepFunction");
		if (stepName.compareTo("Adam") == 0) {
			stepfunction = new AdamStep(properties);
		} else if (stepName.compareTo("Simple") == 0) {
			stepfunction = new GradStep(properties);
		} else if (stepName.compareTo("SimpleL1") == 0) {
			stepfunction = new GradStepL1(properties);			
		} else {
			stepfunction = new AdamStep(properties);
		}
		return stepfunction;
	}
}
