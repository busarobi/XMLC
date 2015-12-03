package preprocessing;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import Data.AVPair;
import Data.AVTable;

public class FeatureHasherTest {

	private AVTable input;

	@Before
	public void setupInputTable() {
		input = new AVTable();
		input.d = 3;
		input.n = 2;
		input.m = 2;
		input.y = new int[][] { { 0, 1 }, { 1, 0 } };

		input.x = new AVPair[][] {
			{ new AVPair(0, 0.1), new AVPair(2, -0.3) },
			{ new AVPair(1, 0.2), new AVPair(2, 0.05) }
		};
	}

	@Test
	public void testTransformRowSparseAVPairArray() {
		FeatureHasher fh = new MurmurHasher(1, 2);

		AVPair[] test = new AVPair[3];
		test[0] = new AVPair(); test[0].index = 0; test[0].value = 0.1;
		test[1] = new AVPair(); test[1].index = 1; test[1].value = 0.3;
		test[2] = new AVPair(); test[2].index = 2; test[2].value = 0.5;
		AVPair[] result = fh.transformRowSparse(test);
		assertEquals(-0.4, result[0].value, 1e-5);
		assertEquals(-0.3, result[1].value, 1e-5);
	}

	@Test
	public void testTransformSparse() {
		FeatureHasher fh = new MurmurHasher(1, 2);
		AVTable result = fh.transformSparse(input);
		assertEquals(0, result.x[0][0].index);
		assertEquals(0.4, result.x[0][0].value, 1e-5);
		assertEquals(0, result.x[1][0].index);
		assertEquals(-0.05, result.x[1][0].value, 1e-5);
		assertEquals(1, result.x[1][1].index);
		assertEquals(-0.2, result.x[1][1].value, 1e-5);
	}

	@Test
	public void testTransformSparseMT() {
		FeatureHasher fh = new MurmurHasher(1, 2, 2);
		AVTable result = fh.transformSparse(input);
		assertEquals(0, result.x[0][0].index);
		assertEquals(-0.7, result.x[0][0].value, 1e-5);
		assertEquals(1, result.x[0][1].index);
		assertEquals(0.1, result.x[0][1].value, 1e-5);
		assertEquals(0, result.x[1][0].index);
		assertEquals(-0.1, result.x[1][0].value, 1e-5);
		assertEquals(1, result.x[1][1].index);
		assertEquals(-0.2, result.x[1][1].value, 1e-5);
	}

}
