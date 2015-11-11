package preprocessing;

import static org.junit.Assert.*;
import org.junit.Test;

import Data.AVPair;
import Data.AVTable;
import preprocessing.FeatureHasher;

public class FeatureHasherTest {

	@Test
	public void testTransformRowSparseAVPairArray() {
		FeatureHasher fh = new FeatureHasher(1, 2);
		
		AVPair[] test = new AVPair[3];
		test[0] = new AVPair(); test[0].index = 0; test[0].value = 0.1;
		test[1] = new AVPair(); test[1].index = 1; test[1].value = 0.3;
		test[2] = new AVPair(); test[2].index = 2; test[2].value = 0.5;
		AVPair[] result = fh.transformRowSparse(test);
		assertEquals(-0.4, result[0].value, 1e-5);
		assertEquals(-0.3, result[1].value, 1e-5);
	}

	/*@Test
	public void testTransformRowSparseAVPairArrayInt() {
		fail("Not yet implemented");
	}*/

	@Test
	public void testTransformSparse() {
		
		FeatureHasher fh = new FeatureHasher(1, 2);
		AVTable input = new AVTable();
		input.d = 3;
		input.n = 2;
		input.m = 2;
		input.y = new int[][] {{0, 1}, {1, 0}};
		input.x = new AVPair[][]{{new AVPair()}, {new AVPair()}};
		input.x[0][0].index = 0; input.x[0][0].value = 0.1;
		input.x[1][0].index = 1; input.x[1][0].value = 0.2;
		
		AVTable result = fh.transformSparse(input);
		assertEquals(0, result.x[0][0].index);
		assertEquals(0.1, result.x[0][0].value, 1e-5);
		assertEquals(1, result.x[1][0].index);
		assertEquals(-0.2, result.x[1][0].value, 1e-5);
	}

}
