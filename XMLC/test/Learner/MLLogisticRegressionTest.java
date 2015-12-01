package Learner;

import static org.junit.Assert.assertEquals;

import java.util.Properties;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import Data.AVPair;
import Data.AVTable;
import Learner.step.GradStep;
import Learner.step.StepFunction;
import jsat.linear.DenseVector;
import util.MasterSeed;


public class MLLogisticRegressionTest {

	AVTable data;
	StepFunction stepFunction;

	@Before
	public void setUp() throws Exception {
		data = new AVTable();
		data.n = 2;
		data.m = 2;
		data.d = 2;
		data.x = new AVPair[][] {
			{ new AVPair(0, 0.1), new AVPair(1, -0.3) },
			{ new AVPair(1, 0.2) }
		};

		stepFunction = new GradStep(1.0);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testUpdatedPosteriors() {
		MasterSeed.setSeed(123);
		MLLogisticRegression learner = new MLLogisticRegression(new Properties(), stepFunction);
		learner.allocateClassifiers(data);
		learner.w = new DenseVector[] {
			new DenseVector(3), new DenseVector(3)
		};
		learner.updatedPosteriors(0, 0, -1.0);
		double[] expected = {0.095, -0.286, 0.953};
		for (int i = 0; i < 3; i++) {
			assertEquals(expected[i], learner.w[0].get(i), 1e-3);
		}

		// Check if weights are updated sparsely
		learner.updatedPosteriors(1, 1, 1.0);
		assertEquals(0.0, learner.w[1].get(0), 1e-3);
	}

}
