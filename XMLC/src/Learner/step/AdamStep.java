package Learner.step;

import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.SparseVectorExt;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.math.Function;

public class AdamStep implements StepFunction {
	private static Logger logger = LoggerFactory.getLogger(AdamStep.class);

	private double T = 1;

	private double beta1;
	private double beta2;
	private double eps;
	private double gamma;

	private SparseVectorExt firstMoments = null;
	private SparseVectorExt secondMoments = null;
	private double bFirst = 0.0;
	private double bSecond = 0.0;

	public AdamStep(Properties properties) {
		logger.info("#####################################################");
		logger.info("#### Optimizer: Adam");
		this.beta1 = Double.parseDouble(properties.getProperty("beta1", "0.9"));
		logger.info("#### beta1: " + this.beta1);
		this.beta2 = Double.parseDouble(properties.getProperty("beta2", "0.999"));
		logger.info("#### beta2: " + this.beta2);
		this.eps = Double.parseDouble(properties.getProperty("eps", "1e-8"));
		logger.info("#### eps: " + this.eps);
		this.gamma = Double.parseDouble(properties.getProperty("gamma", "0.002"));
		logger.info("#### gamma: " + this.gamma);
		logger.info("#####################################################");
	}

	public AdamStep(double gamma, double beta1, double beta2, double eps) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.eps = eps;
		this.gamma = gamma;
	}

	private void allocate(int length) {
		firstMoments = new SparseVectorExt(length);
		secondMoments = new SparseVectorExt(length);
	}

	@Override
	public void step(Vec w, SparseVector grad) {
		this.step(w, grad, 0.0, 0.0);
	}

	@Override
	public double step(Vec w, SparseVector grad, double bias, double biasGrad) {
		if (firstMoments == null) {
			allocate(w.length());
		}
		final double learningRate =
			gamma * Math.sqrt(1.0 - Math.pow(beta2, T)) / (1.0 - Math.pow(beta1, T));
		firstMoments.mutableMultiply(beta1);
		firstMoments.mutableAdd(1.0 - beta1, grad);
		secondMoments.mutableMultiply(beta2);
		Function power2 = new Function() {
			private static final long serialVersionUID = 1L;

			@Override
			public double f(Vec x) { return 0; }

			@Override
			public double f(double... x) {
				return x[0] * x[0];
			}
		};
		grad.applyFunction(power2);
		secondMoments.mutableAdd(1.0 - beta2, grad);
		for (IndexValue iv : firstMoments) {
			final int i = iv.getIndex();
			w.increment(i, -iv.getValue() * learningRate / (Math.sqrt(secondMoments.get(i)) + eps));
		}
		bFirst = beta1 * bFirst + (1.0 - beta1) * biasGrad;
		bSecond = beta2 * bSecond + (1.0 - beta2) * Math.pow(biasGrad, 2.0);
		T++;
		return learningRate * bFirst / (Math.sqrt(bSecond) + eps);
	}

	@Override
	public StepFunction clone() {
		// This currently only clones the parameters but will not copy the internal state.
		StepFunction newstep = new AdamStep(gamma, beta1, beta2, eps);
		return newstep;
	}
	
	public String toString() {
		String r = "";
		final double learningRate =
				gamma * Math.sqrt(1.0 - Math.pow(beta2, T)) / (1.0 - Math.pow(beta1, T));

		r = "Multiplier: " + (learningRate );
		return r;
	}

}
