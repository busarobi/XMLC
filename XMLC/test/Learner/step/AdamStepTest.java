package Learner.step;

import static org.junit.Assert.assertEquals;

import java.util.Properties;
import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import Data.SparseVectorExt;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.math.FunctionVec;
import jsat.math.optimization.RosenbrockFunction;

public class AdamStepTest {

	private Random rand;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
		this.rand = new Random(123);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testStep() {
		// Adapted from JSATs test case
		Vec x0 = new SparseVectorExt(21);
		for (int i = 0; i < 21; i++) {
			x0.set(i, rand.nextDouble());
		}

		RosenbrockFunction f = new RosenbrockFunction();
		FunctionVec fp = f.getDerivative();

		StepFunction sf = new AdamStep(new Properties());
		for (int i = 0; i < 100000; i++) {
			SparseVector grad = (SparseVector) fp.f(x0);
			grad.normalize();
			sf.step(x0, grad);
		}
		assertEquals(0.0, f.f(x0), 1e-1);
	}

}
