package Learner.step;

import jsat.linear.Vec;

/**
 *
 * @author Karlson Pfannschmidt
 */
public interface StepFunction {
	/**
	 * Applies a gradient descent step to the weight vector {@code w}.
	 * @param w weight vector
	 * @param grad vector of gradients
	 * @param bias
	 * @param biasGrad
	 * @return adjustment to bias:
	 * {@code bias = bias - returnValue}
	 */
	public abstract double step(Vec w, Vec grad, double bias, double biasGrad);
}
