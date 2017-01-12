package preprocessing;

import Data.AVPair;
import Data.AVTable;

public class Identical implements FeatureHasher {

	public Identical() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public AVPair[] transformRowSparse(AVPair[] row) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AVPair[] transformRowSparse(AVPair[] row, int taskid) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AVTable transformSparse(AVTable data) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getIndex(int task, int feature) {
		// TODO Auto-generated method stub
		return feature;
	}

	@Override
	public int getSign(int task, int feature) {
		// TODO Auto-generated method stub
		return 1;
	}

}
