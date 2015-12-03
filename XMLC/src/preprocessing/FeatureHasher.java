package preprocessing;

import Data.AVPair;
import Data.AVTable;

public interface FeatureHasher {
	
	public AVPair[] transformRowSparse(AVPair[] row);
	public AVPair[] transformRowSparse(AVPair[] row, int taskid);
	public AVTable transformSparse(AVTable data);
	public int getIndex(int task, int feature);
	
}
