package Learner.step;

import java.util.Properties;

import Data.SparseVector;
import jsat.linear.Vec;

public class AdamStep implements StepFunction {

	private double T = 1;

	private double beta1;
	private double beta2;
	private double eps;
	private double gamma;

	private SparseVector firstMoments;
	private SparseVector secondMoments;
	private double bFirst;
	private double bSecond;

	public AdamStep(Properties properties) {
		System.out.println("#####################################################");
		System.out.println("#### Optimizer: Adam");
		// Exponential decay rates for moment estimates in Adam
		this.beta1 = Double.parseDouble(properties.getProperty("beta1", "0.9"));
		System.out.println("#### beta1: " + this.beta1);
		this.beta2 = Double.parseDouble(properties.getProperty("beta2", "0.999"));
		System.out.println("#### beta2: " + this.beta2);
		this.eps = Double.parseDouble(properties.getProperty("eps", "1e-8"));
		System.out.println("#### eps: " + this.eps);
		this.gamma = Double.parseDouble(properties.getProperty("gamma", "0.002"));
		System.out.println("#### gamma: " + this.gamma);
		System.out.println("#####################################################");

		firstMoments = new SparseVector();
		secondMoments = new SparseVector();
		bFirst = 0.0;
		bSecond = 0.0;
	}

	@Override
	public double step(Vec w, Vec grad, double bias, double biasGrad) {
		final double learningRate =
			gamma * Math.sqrt(1.0 - Math.pow(beta2, T)) / (1.0 - Math.pow(beta1, T));
		firstMoments.mutableMultiply(beta1);
		firstMoments.mutableAdd(grad.multiply(1.0 - beta1));
		secondMoments.mutableMultiply(beta2);
		secondMoments.mutableAdd(grad.pairwiseMultiply(grad).multiply(1.0 - beta2));
		w.mutableSubtract(firstMoments.multiply(learningRate).pairwiseDivide(secondMoments.sqrt().add(eps)));
		bFirst = beta1 * bFirst + (1.0 - beta1) * biasGrad;
		bSecond = beta2 * bSecond + (1.0 - beta2) * Math.pow(biasGrad, 2.0);
		T++;
		return learningRate * bFirst / (Math.sqrt(bSecond) + eps);
	}

}
