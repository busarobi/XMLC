package Learner.step;

import java.util.Properties;

import jsat.linear.SubVector;
import jsat.linear.Vec;

public class GradStep implements StepFunction {

	private double learningRate;

	public GradStep(Properties properties) {
		System.out.println("#####################################################");
		System.out.println("#### Optimizer: Simple gradient descent");
		this.learningRate = Double.parseDouble(properties.getProperty("gamma", "0.002"));
		System.out.println("#### gamma: " + this.learningRate);
		System.out.println("#####################################################");
	}

	public GradStep(double gamma) {
		this.learningRate = gamma;
	}

	@Override
	public void step(Vec w, Vec grad) {
		int n = w.length();
		double biasAdj = this.step(w, new SubVector(0, n - 1, grad),
			                       w.get(n - 1), grad.get(n - 1));
		w.set(n - 1, w.get(n - 1) - biasAdj);
	}

	@Override
	public double step(Vec w, Vec grad, double bias, double biasGrad) {
		w.mutableSubtract(grad.multiply(learningRate));
		return learningRate * biasGrad;
	}

	@Override
	public StepFunction clone() {
		StepFunction newstep = new GradStep(this.learningRate);
		return newstep;
	}

}
