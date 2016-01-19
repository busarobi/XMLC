package preprocessing;

import java.util.ArrayList;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import util.HashFunction;

public class MurmurHasher implements FeatureHasher {

	private static Logger logger = LoggerFactory.getLogger(MurmurHasher.class);

	private HashFunction hash;
	private HashFunction sign;
	private HashFunction[] taskhash;
	private HashFunction[] tasksign;
	private boolean isMultiTask = false;
	private int nFeatures;
	private int nTasks;

	public MurmurHasher(int seed, int nFeatures) {
		
		this.nFeatures = nFeatures;
		this.hash = new HashFunction(seed, nFeatures);
		this.sign = new HashFunction(seed + 1);
		
		logger.info("#####################################################" );
		logger.info("#### Murmur hash" );
		logger.info("#### Num. of hashed features: " + this.nFeatures );
		logger.info("#####################################################" );
	}

	public MurmurHasher(int seed, int nFeatures, int nTasks) {
		this(seed, nFeatures);
		this.isMultiTask = true;
		this.nTasks = nTasks;
		this.taskhash = new HashFunction[nTasks];
		this.tasksign = new HashFunction[nTasks];
		for (int i = 0; i < nTasks; i++) {
			this.taskhash[i] = new HashFunction(seed + i, nFeatures);
			this.tasksign[i] = new HashFunction(seed + nTasks + i);
		}
	}

	public AVPair[] transformRowSparse(AVPair[] row) throws IllegalStateException {
		if (this.isMultiTask) {
			throw new IllegalArgumentException("In the multitask setting a task id is necessary.");
		}
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

	private AVPair[] convertHashMap(HashMap<Integer, Double> hm) {
		ArrayList<AVPair> result = new ArrayList<AVPair>(hm.size());
		for (Integer key : hm.keySet()) {
			if (Math.abs(hm.get(key)) > 1e-10) {
				AVPair pair = new AVPair();
				pair.index = key;
				pair.value = hm.get(key);
				result.add(pair);
			}
		}
		return result.toArray(new AVPair[result.size()]);
	}

	public AVPair[] transformRowSparse(AVPair[] row, int taskid) {
		if (!this.isMultiTask) {
			throw new IllegalArgumentException("The task id is only used in multi-task feature hashing");
		}
		HashMap<Integer, Double> sums = new HashMap<Integer, Double>(this.nFeatures);
		for (AVPair pair : row) {
			int hi = this.taskhash[taskid].hash(pair.index);
			double val = sums.containsKey(hi) ? sums.get(hi) : 0.0;
			sums.put(hi, val + pair.value * this.tasksign[taskid].hash(pair.index));
		}
		return this.convertHashMap(sums);
	}

	public AVTable transformSparse(AVTable data) {
		AVTable result = new AVTable();
		result.n = data.n;
		result.m = data.m;
		result.d = this.nFeatures;
		result.y = data.y;
		result.x = new AVPair[result.n][];
		if (this.isMultiTask) {
			for (int row = 0; row < result.n; row++) {
				HashMap<Integer, Double> sums = new HashMap<Integer, Double>(this.nFeatures);
				for (int t = 0; t < this.nTasks; t++) {
					for (AVPair pair : data.x[row]) {
						int hi = this.taskhash[t].hash(pair.index);
						double val = sums.containsKey(hi) ? sums.get(hi) : 0.0;
						sums.put(hi, val + pair.value * this.tasksign[t].hash(pair.index));
					}
				}
				result.x[row] = convertHashMap(sums);
			}
		} else {
			for (int i = 0; i < result.n; i++) {
				result.x[i] = this.transformRowSparse(data.x[i]);
			}
		}
		return result;
	}

	public int getSign( int label, int feature ) {
		int value = (label<<1-1)*feature; 
		return ((value & 1) == 0) ? -1 : 1;
	}
	
//	public int getSign( int task, int feature ) {
//		return this.tasksign[task].hash(feature);
//	}
	
	public int getIndex(int task, int feature) {
		return this.taskhash[task].hash(feature);
	}
	
	
	public static void main(String[] args) {
		AVPair[] test = new AVPair[3];
		test[0] = new AVPair(); test[0].index = 0; test[0].value = 0.1;
		test[1] = new AVPair(); test[1].index = 1; test[1].value = 0.3;
		test[2] = new AVPair(); test[2].index = 2; test[2].value = 0.5;
		MurmurHasher fh = new MurmurHasher(0, 2);
		AVPair[] result = fh.transformRowSparse(test);
		for (AVPair p : result) {
			logger.info("Index: " + p.index + " Value: " + p.value);
		}
		logger.info("");

		test = new AVPair[3];
		test[0] = new AVPair();
		test[0].index = 0;
		test[0].value = 0.1;
		test[1] = new AVPair();
		test[1].index = 1;
		test[1].value = 0.3;
		test[2] = new AVPair();
		test[2].index = 2;
		test[2].value = 0.5;
		int nTasks = 3;
		fh = new MurmurHasher(0, 4, nTasks);
		result = fh.transformRowSparse(test, 1);
		for (int i = 0; i < nTasks; i++) {
			logger.info("Task #" + i + ":");
			result = fh.transformRowSparse(test, i);
			for (AVPair p : result) {
				logger.info("Index: " + p.index + " Value: " + p.value);
			}
		}
		logger.info("");

		fh = new MurmurHasher(1, 2, 2);
		AVTable input = new AVTable();
		input.d = 3;
		input.n = 2;
		input.m = 2;
		input.y = new int[][] { { 0, 1 }, { 1, 0 } };
		input.x = new AVPair[][] {
			{ new AVPair(0, 0.1), new AVPair(2, -0.3) },
			{ new AVPair(1, 0.2), new AVPair(2, 0.05) }
		};
		AVTable cv_table = fh.transformSparse(input);
		for (int i = 0; i < cv_table.n; i++) {
			for (int j = 0; j < cv_table.x[i].length; j++) {
				logger.info("Index: " + cv_table.x[i][j].index + " Value: " + cv_table.x[i][j].value);
			}
		}
	}

}
