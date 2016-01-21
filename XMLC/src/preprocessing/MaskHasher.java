package preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;

public class MaskHasher implements FeatureHasher {
	private static Logger logger = LoggerFactory.getLogger(MaskHasher.class);


	private long nFeatures;
	private long nTasks;
	private long mask = 1;	
	
	public MaskHasher( int seed, int nFeatures, int nTasks ) {
		this.nFeatures = nFeatures;
		this.nTasks = nTasks;
				
		this.mask = this.nFeatures - 1;
		
		logger.info("#####################################################" );
		logger.info("#### Mask hash" );
		logger.info("#### Num. of hashed features: " + (nTasks * this.nFeatures) );
		logger.info("#####################################################" );
		
	}

//	public int getIndex(int label, int feature) {
//		return ((feature * this.nTasks + label)) & (this.mask); 
//	}

	public int getIndex(int label, int feature) {
		return (int) ((((feature * this.nTasks) + label)) & (this.mask)); 
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
