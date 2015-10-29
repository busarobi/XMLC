package Learner;

import Data.AVTable;

public abstract class AbstractLearner {

	public abstract void allocateClassifiers( AVTable data );
	public abstract void train( AVTable data );
	public abstract void test( AVTable data );
}
