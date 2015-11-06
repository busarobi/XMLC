package preprocessing;

import java.util.HashMap;

import Data.AVPair;
import Data.AVTable;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import util.HashFunction;

public class FeatureHasher {
	

	private HashFunction hash;
	private HashFunction sign;
	private HashFunction[] taskhash;
	private boolean isMultiTask = false;
	private int nFeatures;
	private boolean withSharedTask;

	public FeatureHasher(int seed, int nFeatures) {
		this.nFeatures = nFeatures;
		this.hash = new HashFunction(seed, nFeatures);
		this.sign = new HashFunction(seed + 1);
	}
	
	public FeatureHasher(int seed, int nFeatures, int nTasks,
			             boolean sharedTask) {
		this(seed, nFeatures);
		this.isMultiTask = true;
		this.withSharedTask = sharedTask;
		this.taskhash = new HashFunction[nTasks];
		for (int i = 0; i < nTasks; i++) {
			this.taskhash[i] = new HashFunction(seed + i + 2, nFeatures); 
		}
	}
	
	public double[] transformRow(AVPair[] row) {
		int nPairs = row.length;
		double[] result = new double[nPairs];
		//TODO
		return result;
	}
	
	public AVPair[] transformRowSparse(AVPair[] row) throws IllegalStateException {
		if (this.isMultiTask)
			throw new IllegalStateException("In the multitask setting this a task index is necessary.");
		HashMap<Integer, Double> sums = new HashMap<Integer, Double>(this.nFeatures);
		for (AVPair pair : row) {
			int hi = this.hash.hash(pair.index);
			double val = sums.containsKey(hi) ? sums.get(hi) : 0.0;
			sums.put(hi, val + pair.value * this.sign.hash(pair.index));
		}
		AVPair[] result = new AVPair[sums.size()];
		int i = 0;
		for (Integer key : sums.keySet()) {
			result[i] = new AVPair();
			result[i].index = key;
			result[i].value = sums.get(key);
			i++;
		}
		return result;
	}
	
	public AVPair[] transformRowSparse(AVPair[] row, int taskid) {
		if (true)
			throw new NotImplementedException();
		if (!this.isMultiTask)
			return transformRowSparse(row);
		//TODO:
		HashMap<Integer, Double> sums = new HashMap<Integer, Double>(this.nFeatures);
		for (AVPair pair : row) {
			int hi = this.hash.hash(pair.index);
			double val = sums.containsKey(hi) ? sums.get(hi) : 0.0;
			sums.put(hi, val + pair.value * this.sign.hash(pair.index));
		}
		AVPair[] result = new AVPair[sums.size()];
		int i = 0;
		for (Integer key : sums.keySet()) {
			result[i] = new AVPair();
			result[i].index = key;
			result[i].value = sums.get(key);
			i++;
		}
		return result;
	}
	
	public double[][] transform(AVTable data) {
		double[][] result = new double[data.n][data.d];
		for (int i = 0; i < data.n; i++) {
			result[i] = this.transformRow(data.x[i]);
		}
		return result;
	}
	
	public AVTable transformSparse(AVTable data) {
		AVTable result = new AVTable();
		result.n = data.n;
		result.m = data.m;
		result.d = this.nFeatures;
		result.y = data.y;
		result.x = new AVPair[result.n][];
		for (int i = 0; i < result.n; i++) {
			result.x[i] = this.transformRowSparse(data.x[i]);
		}		
		return result;
	}

	public static void main(String[] args) {
		AVPair[] test = new AVPair[3];
		test[0] = new AVPair(); test[0].index = 0; test[0].value = 0.1;
		test[1] = new AVPair(); test[1].index = 1; test[1].value = 0.3;
		test[2] = new AVPair(); test[2].index = 2; test[2].value = 0.5;
		FeatureHasher fh = new FeatureHasher(0, 2);
		AVPair[] result = fh.transformRowSparse(test);
		for (AVPair p : result) {
			System.out.println("Index: " + p.index + " Value: " + p.value);
		}
	}

}
