package threshold;

import static org.junit.Assert.*;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import util.Constants;

public class OfoFastThresholdTunerTests {
	OfoFastThresholdTuner target;
	final int totalNumberOfLabels = 50000;
	final int maximumNumberOfLabelsPerInstance = 5;
	private static DecimalFormat df2 = new DecimalFormat(".##");

	@Before
	public void arrange() {
		target = new OfoFastThresholdTuner(totalNumberOfLabels, null);
	}

	@After
	public void tearDown() {
		target = null;
	}

	private List<Integer> getRandomInts() {
		return getRandomInts(maximumNumberOfLabelsPerInstance, totalNumberOfLabels);
	}

	private List<Integer> getRandomInts(int maximumNumberOfLabelsPerInstance, int totalNumberOfLabels) {

		int numberOfLabels = ThreadLocalRandom.current()
				.nextInt(1, maximumNumberOfLabelsPerInstance + 1);

		ArrayList<Integer> retVal = new ArrayList<>();

		for (int i = 0; i < numberOfLabels; i++)
			retVal.add(ThreadLocalRandom.current()
					.nextInt(totalNumberOfLabels));

		return retVal;
	}

	// START benchmark tests
	/*
	 * // To run below tests on performance benchmark on
	 * tuneAndGetAffectedLabels // and tuneAndGetAffectedLabels1, // change the
	 * access modifiers of those.
	 * 
	 * @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_SingleInstances() {
	 * // Arrange int numberOfInstances = 1; double iteration = 100000000;
	 * double elapsed1 = 0, elapsed2 = 0;
	 * 
	 * for (int k = 0; k < iteration; k++) { List<HashSet<Integer>> trueLabels =
	 * new ArrayList<HashSet<Integer>>(); List<HashSet<Integer>> predictedLabels
	 * = new ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act
	 * 
	 * long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; }
	 * 
	 * // Assert System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances10
	 * () { // Arrange int numberOfInstances = 10; double iteration = 1000000;
	 * double elapsed1 = 0, elapsed2 = 0; for (int k = 0; k < iteration; k++) {
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act
	 * 
	 * long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances100
	 * () { // Arrange int numberOfInstances = 100; double iteration = 100;
	 * 
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act double elapsed1 = 0, elapsed2 = 0; for (int i = 0; i < iteration;
	 * i++) { long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances1K
	 * () { // Arrange int numberOfInstances = 1000; double iteration = 100;
	 * double elapsed1 = 0, elapsed2 = 0; for (int k = 0; k < iteration; k++) {
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act
	 * 
	 * long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances10K
	 * () { // Arrange int numberOfInstances = 10000; double iteration = 1000;
	 * double elapsed1 = 0, elapsed2 = 0; for (int k = 0; k < iteration; k++) {
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act
	 * 
	 * long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * // @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances100k
	 * () { // Arrange int numberOfInstances = 100000; double iteration = 100;
	 * double elapsed1 = 0, elapsed2 = 0; for (int k = 0; k < iteration; k++) {
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act
	 * 
	 * long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 * 
	 * // @Test public void
	 * performanceComparisonBetweenSetBasedTuningAndOriginal_NumberOfInstances1M
	 * () { // Arrange int numberOfInstances = 1000000; double iteration = 100;
	 * 
	 * List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
	 * List<HashSet<Integer>> predictedLabels = new
	 * ArrayList<HashSet<Integer>>();
	 * 
	 * for (int i = 0; i < numberOfInstances; i++) { trueLabels.add(new
	 * HashSet<Integer>(getRandomInts())); predictedLabels.add(new
	 * HashSet<Integer>(getRandomInts())); }
	 * 
	 * // act double elapsed1 = 0, elapsed2 = 0; for (int i = 0; i < iteration;
	 * i++) { long startTime1 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels(predictedLabels, trueLabels); elapsed1 +=
	 * (double) (System.nanoTime() - startTime1) / iteration;
	 * 
	 * long startTime2 = System.nanoTime();
	 * target.tuneAndGetAffectedLabels1(predictedLabels, trueLabels); elapsed2
	 * += (double) (System.nanoTime() - startTime2) / iteration; } // Assert
	 * System.out.println("Average (over " + iteration +
	 * " iterations) time taken for " + numberOfInstances +
	 * " instances by the original method : " + df2.format(elapsed1 / (1000.0 *
	 * numberOfInstances)) + " ms."); System.out.println("Average  (over " +
	 * iteration + " iterations) time taken for " + numberOfInstances +
	 * " instances by the set based  method : " + df2.format(elapsed2 / (1000.0
	 * * numberOfInstances)) + " ms."); assert (true);// nothing to assert }
	 */
	// END benchmark tests

	@Test
	public void getTunedThresholds_ReturnsValue_EvenInAbsenceOfTuningData() {
		// Arrange
		int totalNumberOfLabels = 10;
		target = new OfoFastThresholdTuner(totalNumberOfLabels, null);
		double[] expected = new double[totalNumberOfLabels];
		Arrays.fill(expected, (1 / 100.0));

		// act
		double[] actual = target.getTunedThresholds(null);

		// Assert
		assertArrayEquals(expected, actual, 0.0001);
	}

	@Test
	public void getTunedThresholds_ReturnsValueAsPerSeedValue_EvenInAbsenceOfTuningData() {
		// Arrange
		int totalNumberOfLabels = 10;
		ThresholdTunerInitOption options = new ThresholdTunerInitOption() {
			{
				aSeed = 3;
			}
		};
		target = new OfoFastThresholdTuner(totalNumberOfLabels, options);
		double[] expected = new double[totalNumberOfLabels];
		Arrays.fill(expected, (3 / 100.0));

		// act
		double[] actual = target.getTunedThresholds(null);

		// Assert
		assertArrayEquals(expected, actual, 0.0001);
	}

	@SuppressWarnings("serial")
	@Test
	public void getTunedThresholds_TrueAndPredictedLabelSetAreSame_ReturnsCorrectValue() {
		// Arrange
		int totalNumberOfLabels = 10;
		ThresholdTunerInitOption options = new ThresholdTunerInitOption() {
			{
				aSeed = 3;
				bSeed = 80;
			}
		};
		target = new OfoFastThresholdTuner(totalNumberOfLabels, options);

		final List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				// add(new HashSet<Integer>(Arrays.asList(2, 3, 5, 8, 9)));
			}
		};
		final List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				// add(new HashSet<Integer>(Arrays.asList(0, 2, 3, 5, 8)));
			}
		};
		Map<String, Object> tuningData = new HashMap<String, Object>() {
			{
				put(Constants.ThresholdTuningDictKeys.trueLabels, trueLabels);
				put(Constants.ThresholdTuningDictKeys.predictedLabels, predictedLabels);
			}
		};
		double[] expected = { 3.0 / 80.0, 4.0 / 82.0, 4.0 / 82.0, 4.0 / 82.0, 3.0 / 80.0, 3.0 / 80.0, 3.0 / 80.0,
				3.0 / 80.0, 3.0 / 80.0, 3.0 / 80.0 };

		// act
		double[] actual = target.getTunedThresholds(tuningData);

		// Assert
		assertArrayEquals(expected, actual, 0.0001);
	}

	@SuppressWarnings("serial")
	@Test
	public void getTunedThresholds_TrueAndPredictedLabelSetAreNotSame_ReturnsCorrectValue() {
		// Arrange
		int totalNumberOfLabels = 10;
		ThresholdTunerInitOption options = new ThresholdTunerInitOption() {
			{
				aSeed = 3;
				bSeed = 80;
			}
		};
		target = new OfoFastThresholdTuner(totalNumberOfLabels, options);

		final List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(2, 3, 5, 8, 9)));
				add(new HashSet<Integer>(Arrays.asList(7, 8)));
			}
		};
		final List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(0, 2, 3, 5, 8)));
				add(new HashSet<Integer>(Arrays.asList(1, 6, 9)));
			}
		};
		Map<String, Object> tuningData = new HashMap<String, Object>() {
			{
				put(Constants.ThresholdTuningDictKeys.trueLabels, trueLabels);
				put(Constants.ThresholdTuningDictKeys.predictedLabels, predictedLabels);
			}
		};
		double[] expected = { 3.0 / 81.0, 4.0 / 83.0, 5.0 / 84.0, 5.0 / 84.0, 3.0 / 80.0, 4.0 / 82.0, 3.0 / 81.0,
				3.0 / 81.0, 4.0 / 83.0, 3.0 / 82.0 };

		// act
		double[] actual = target.getTunedThresholds(tuningData);

		// Assert
		assertArrayEquals(expected, actual, 0.0001);
	}

	@Test(expected = IllegalArgumentException.class)
	public void getTunedThresholdsSparse_ThrowsException_InAbsenceOfTuningData() throws Exception {

		target.getTunedThresholdsSparse(null);

	}

	@Test(expected = IllegalArgumentException.class)
	public void getTunedThresholdsSparse_ThrowsException_InAbsenceOfPredictedLabels() throws Exception {

		// arrange
		final List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(2, 3, 5, 8, 9)));
				add(new HashSet<Integer>(Arrays.asList(7, 8)));
			}
		};
		Map<String, Object> tuningData = new HashMap<String, Object>() {
			{
				put(Constants.ThresholdTuningDictKeys.trueLabels, trueLabels);
			}
		};

		// act
		target.getTunedThresholdsSparse(tuningData);
	}

	@Test(expected = IllegalArgumentException.class)
	public void getTunedThresholdsSparse_ThrowsException_InAbsenceOfTrueLabels() throws Exception {

		// arrange
		final List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(0, 2, 3, 5, 8)));
				add(new HashSet<Integer>(Arrays.asList(1, 6, 9)));
			}
		};
		Map<String, Object> tuningData = new HashMap<String, Object>() {
			{
				put(Constants.ThresholdTuningDictKeys.predictedLabels, predictedLabels);
			}
		};

		// act
		target.getTunedThresholdsSparse(tuningData);
	}

	@Test
	public void getTunedThresholdsSparse_ReturnsCorrectLabels() throws Exception {

		// arrange
		final List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(2, 3, 5, 8, 9)));
				add(new HashSet<Integer>(Arrays.asList(7, 8)));
			}
		};
		final List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>() {
			{
				add(new HashSet<Integer>(Arrays.asList(1, 2, 3)));
				add(new HashSet<Integer>(Arrays.asList(0, 2, 3, 5, 8)));
				add(new HashSet<Integer>(Arrays.asList(1, 6, 9)));
			}
		};
		Map<String, Object> tuningData = new HashMap<String, Object>() {
			{
				put(Constants.ThresholdTuningDictKeys.trueLabels, trueLabels);
				put(Constants.ThresholdTuningDictKeys.predictedLabels, predictedLabels);
			}
		};
		Set<Integer> expected = new HashSet(Arrays.asList(0, 1, 2, 3, 5, 6, 7, 8, 9));

		// act
		Map<Integer, Double> sparseThresholds = target.getTunedThresholdsSparse(tuningData);
		Set<Integer> actual = sparseThresholds.keySet();

		// assert
		assertTrue(expected.equals(actual));
	}

	@Test
	public void getTunedThresholdsSparse_ReturnsCorrectLabels_For100RandomInstances() throws Exception {

		// arrange
		int numberOfInstances = 100;
		List<HashSet<Integer>> trueLabels = new ArrayList<HashSet<Integer>>();
		List<HashSet<Integer>> predictedLabels = new ArrayList<HashSet<Integer>>();

		for (int i = 0; i < numberOfInstances; i++) {
			trueLabels.add(new HashSet<Integer>(getRandomInts()));
			predictedLabels.add(new HashSet<Integer>(getRandomInts()));
		}

		Map<String, Object> tuningData = new HashMap<String, Object>();
		tuningData.put(Constants.ThresholdTuningDictKeys.trueLabels, trueLabels);
		tuningData.put(Constants.ThresholdTuningDictKeys.predictedLabels, predictedLabels);
		Set<Integer> expected = new HashSet<Integer>();
		for (HashSet<Integer> labels : trueLabels) {
			expected.addAll(labels);
		}
		for (HashSet<Integer> labels : predictedLabels) {
			expected.addAll(labels);
		}

		// act
		Map<Integer, Double> sparseThresholds = target.getTunedThresholdsSparse(tuningData);
		Set<Integer> actual = sparseThresholds.keySet();

		// assert
		assertTrue(expected.equals(actual));
	}

}
