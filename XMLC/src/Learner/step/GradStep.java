package Learner.step;

import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jsat.linear.SparseVector;
import jsat.linear.Vec;

public class GradStep implements StepFunction {
	private static Logger logger = LoggerFactory.getLogger(GradStep.class);

	private double learningRate;
	private int T = 0;
	private int step = 1000;
	
	public GradStep(Properties properties) {
		logger.info("#####################################################");
		logger.info("#### Optimizer: Simple gradient descent");
		
		this.learningRate = Double.parseDouble(properties.getProperty("gamma", "0.002"));
		logger.info("#### gamma: " + this.learningRate);

		this.step = Integer.parseInt(properties.getProperty("step", "1000"));
		logger.info("#### step: " + this.step );		
		
		logger.info("#####################################################");
	}

	public GradStep(double gamma) {
		this.learningRate = gamma;
	}

	@Override
	public void step(Vec w, SparseVector grad) {
		this.step(w, grad, 0.0, 0.0);
	}

	@Override
	public double step(Vec w, SparseVector grad, double bias, double biasGrad) {
		this.T++;
		double mult = 1.0 / (Math.ceil(this.T / ((double) this.step)));
		
		w.mutableSubtract(grad.multiply(learningRate * mult));
		return learningRate * mult * biasGrad;
	}

	@Override
	public StepFunction clone() {
		StepFunction newstep = new GradStep(this.learningRate);
		return newstep;
	}

	public String toString() {
		String r = "";
		double mult = 1.0 / (Math.ceil(this.T / ((double) this.step)));
		r = "Multiplier: " + (learningRate * mult);
		return r;
	}
	
}
