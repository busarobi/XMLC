package preprocessing;

import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;

public class UniversalHasher implements FeatureHasher {
	private static Logger logger = LoggerFactory.getLogger(UniversalHasher.class);


	private int nFeatures;
	private int nTasks;
	private int prime = 1;
	private int b = 1;
	private int a = 1;
	private int c = 1;
	private int d = 1;
	
	private int shift = 0;
	private int multiplier = 1;
	private int add = 0;
	private final int INTEGER_LENGHT = 32;
	
	public UniversalHasher(int seed, int nFeatures, int nTasks) {
		
		
		this.nFeatures = nFeatures;
		this.nTasks = nTasks;
		
		this.shift = INTEGER_LENGHT - Integer.numberOfTrailingZeros(nFeatures);
		logger.info("Shift: " + this.shift);
		this.multiplier = (this.nTasks & 1) == 0 ? this.nTasks +1 : this.nTasks; //<<1) + 1;
		logger.info("Shift: " + this.shift + " Multiplier: " + this.multiplier);
		
		Random random = new Random(seed);
		
		this.add = random.nextInt(1 << this.shift);
		
		this.prime = nextprime(this.nFeatures);
		logger.info("Prime: " + this.prime);
		
		this.a = (2*random.nextInt()-1) % this.prime;
		this.b = random.nextInt() % this.prime;

		this.c = (2*random.nextInt()-1) % this.prime;
		this.d = random.nextInt() % this.prime;		
		
		logger.info("#####################################################" );
		logger.info("#### Universal hash" );
		logger.info("#### Num. of hashed features: " + (nTasks * this.nFeatures) );
		logger.info("#####################################################" );
		
	}

	boolean is_prime(int perhapsprime) {

	    int limit;
	    int testfactor;

	    limit = (perhapsprime >> 1) + 1; /** SEE NOTES!!! */

	    for (testfactor = 3; testfactor <= limit; ++testfactor) {
	        if ((perhapsprime % testfactor) == 0) /* If testfactor divides perhapsprime... */
	        {
	            return false; /* ...then, perhapsprime was non-prime. */
	        }
	    }

	    return true;
	}

	
	int nextprime(int inval) {
	    int perhapsprime; /* Holds a tentative prime while we check it. */
	    boolean found;

	    if (inval < 2) /* Initial sanity check of parameter. */
	    {
	        return (2); /* Easy special case. */
	    }

	    /* Testing an even number for primeness is pointless, since
	     * all even numbers are divisible by 2. Therefore, we make sure
	     * that perhapsprime is larger than the parameter, and odd. */
	    perhapsprime = (inval + 1) | 1;

	    perhapsprime -= 2; /* pre-set the loop variable */
	    do {
	        perhapsprime += 2;
	        found = is_prime(perhapsprime);
	    } while (found != true);
	    return perhapsprime;
	}

	

	public int getIndex(int label, int feature) {
		return Math.abs(Math.abs((feature*this.nTasks + label) + this.b) % this.prime) % this.nFeatures;
	}

	


	public int getSign( int label, int feature ) {
		int value = (label<<1-1)*feature; 
		return ((value & 1) == 0) ? -1 : 1;
	}
	
// old universal hashing for signs  	
//	public int getSign(int label, int feature) {
		//int value = (Math.abs((feature*this.nTasks + label) + this.d) % this.prime);// % this.nFeatures;
		//return ((value & 1) == 0) ? -1 : 1;
//	}	
		
	
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
