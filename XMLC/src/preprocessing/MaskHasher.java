package preprocessing;

import java.util.Random;

import Data.AVPair;
import Data.AVTable;

public class MaskHasher implements FeatureHasher {

	private int nFeatures;
	private int nTasks;
	private int mask = 1;	
	
	public MaskHasher( int seed, int nFeatures, int nTasks ) {
		this.nFeatures = nFeatures;
		this.nTasks = nTasks;
				
		this.mask = this.nFeatures - 1;
		
		System.out.println("#####################################################" );
		System.out.println("#### Mask hash" );
		System.out.println("#### Num. of hashed features: " + (nTasks * this.nFeatures) );
		System.out.println("#####################################################" );
		
	}

	public int getIndex(int label, int feature) {
		return ((feature * this.nTasks + label)) & (this.mask); 
	}

	
	public int getSign(int label, int feature) {
		return  ( ( ( (label<<1-1)*feature) & 1) << 1) - 1;
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
	
	
}
