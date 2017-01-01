package threshold;

import static org.junit.Assert.*;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class OfoFastThresholdTunerTests {
	OfoFastThresholdTuner target;
	int numberOfLabels = 50000;
	int maximumNumberOfLabelsPerInstance = 5;
	private static DecimalFormat df2 = new DecimalFormat(".##");

	@Before
	public void arrange() {
		target = new OfoFastThresholdTuner(numberOfLabels, null);
	}

	@After
	public void tearDown() {
		target = null;
	}

	private List<Integer> getRandomInts() {

		int numberOfLabels = ThreadLocalRandom.current()
				.nextInt(1, maximumNumberOfLabelsPerInstance + 1);

		ArrayList<Integer> retVal = new ArrayList<>();

		for (int i = 0; i < numberOfLabels; i++)
			retVal.add(ThreadLocalRandom.current()
					.nextInt(numberOfLabels));

		return retVal;
	}

	// START performance tests
	// To run below tests on performance benchmark on tuneAndGetAffectedLabels and tuneAndGetAffectedLabels1,
	// change the access modifiers of those.
/*
	@Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_SingleInstances() {
		// Arrange
		int numberOfInstances = 1;
		double iteration = 100000000;
		double elapsed1 = 0, elapsed2 = 0;

		for (int k = 0; k < iteration; k++) {
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			for (int i = 0; i < numberOfInstances; i++) {
				trueLabels.add(new HashSet<Integer>(getRandomInts()));
				predictedLabels.add(new HashSet<Integer>(getRandomInts()));
			}

			// act

			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}

		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	@Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances10() {
		// Arrange
		int numberOfInstances = 10;
		double iteration = 1000000;
		double elapsed1 = 0, elapsed2 = 0;
		for (int k = 0; k < iteration; k++) {
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			for (int i = 0; i < numberOfInstances; i++) {
				trueLabels.add(new HashSet<Integer>(getRandomInts()));
				predictedLabels.add(new HashSet<Integer>(getRandomInts()));
			}

			// act

			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	@Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances100() {
		// Arrange
		int numberOfInstances = 100;
		double iteration = 100;

		List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
		List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

		for (int i = 0; i < numberOfInstances; i++) {
			trueLabels.add(new HashSet<Integer>(getRandomInts()));
			predictedLabels.add(new HashSet<Integer>(getRandomInts()));
		}

		// act
		double elapsed1 = 0, elapsed2 = 0;
		for (int i = 0; i < iteration; i++) {
			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	@Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances1K() {
		// Arrange
		int numberOfInstances = 1000;
		double iteration = 100;
		double elapsed1 = 0, elapsed2 = 0;
		for (int k = 0; k < iteration; k++) {
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			for (int i = 0; i < numberOfInstances; i++) {
				trueLabels.add(new HashSet<Integer>(getRandomInts()));
				predictedLabels.add(new HashSet<Integer>(getRandomInts()));
			}

			// act

			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	@Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances10K() {
		// Arrange
		int numberOfInstances = 10000;
		double iteration = 1000;
		double elapsed1 = 0, elapsed2 = 0;
		for (int k = 0; k < iteration; k++) {
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			for (int i = 0; i < numberOfInstances; i++) {
				trueLabels.add(new HashSet<Integer>(getRandomInts()));
				predictedLabels.add(new HashSet<Integer>(getRandomInts()));
			}

			// act

			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	// @Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances100k() {
		// Arrange
		int numberOfInstances = 100000;
		double iteration = 100;
		double elapsed1 = 0, elapsed2 = 0;
		for (int k = 0; k < iteration; k++) {
			List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
			List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

			for (int i = 0; i < numberOfInstances; i++) {
				trueLabels.add(new HashSet<Integer>(getRandomInts()));
				predictedLabels.add(new HashSet<Integer>(getRandomInts()));
			}

			// act

			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}

	// @Test
	public void performanceComparisonBetweenSetBasedTuningAndOldSchool_NumberOfInstances1M() {
		// Arrange
		int numberOfInstances = 1000000;
		double iteration = 100;

		List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
		List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

		for (int i = 0; i < numberOfInstances; i++) {
			trueLabels.add(new HashSet<Integer>(getRandomInts()));
			predictedLabels.add(new HashSet<Integer>(getRandomInts()));
		}

		// act
		double elapsed1 = 0, elapsed2 = 0;
		for (int i = 0; i < iteration; i++) {
			long startTime1 = System.nanoTime();
			target.tuneAndGetAffectedLabels(predictedLabels, trueLabels);
			elapsed1 += (double) (System.nanoTime() - startTime1) / iteration;

			long startTime2 = System.nanoTime();
			target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels);
			elapsed2 += (double) (System.nanoTime() - startTime2) / iteration;
		}
		// Assert
		System.out.println("Average (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the original method : "
				+ df2.format(elapsed1 / (1000.0 * numberOfInstances)) + " ms.");
		System.out.println("Average  (over " + iteration + " iterations) time taken for " + numberOfInstances
				+ " instances by the set based  method : "
				+ df2.format(elapsed2 / (1000.0 * numberOfInstances)) + " ms.");
		assert (true);// nothing to assert
	}
*/
	// END performance tests
}
