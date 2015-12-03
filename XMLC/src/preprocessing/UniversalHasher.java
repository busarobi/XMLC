package preprocessing;

import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import util.HashFunction;

public class UniversalHasher implements FeatureHasher {


	private int seed = 0;
	
	private HashFunction[] taskhash;
	private HashFunction[] tasksign;
	private int nFeatures;
	private int nTasks;
	private int prime = 1;
	private int b = 1;
	private int a = 1;
	
	public UniversalHasher(int seed, int nFeatures, int nTasks) {
		
		
		this.seed = seed;
		this.nFeatures = nFeatures;
		this.nTasks = nTasks;
		this.taskhash = new HashFunction[nTasks];
		this.tasksign = new HashFunction[nTasks];
		
		Random random = new Random(seed);
		
		for (int i = 0; i < nTasks; i++) {
			this.taskhash[i] = new HashFunction(seed + i, nFeatures);
			this.tasksign[i] = new HashFunction(seed*nTasks + i);
		}
		
		this.prime = nextprime(this.nFeatures);
		System.out.println("Prime: " + this.prime);
		
		this.a = (2*random.nextInt()-1) % this.prime;
		this.b = random.nextInt() % this.prime;

		System.out.println("#####################################################" );
		System.out.println("#### Murmur hash" );
		System.out.println("#### Num. of hashed features: " + this.nFeatures );
		System.out.println("#####################################################" );
		
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
		//return (Math.abs((feature*this.nTasks + label) + this.b) % this.prime) % this.nFeatures;
		return Math.abs(Math.abs((feature*this.nTasks + label) + this.b) % this.prime) % this.nFeatures;
		
		//return (Math.abs((feature*this.nTasks + label))) % this.nFeatures;
		//return this.taskhash[label].hash(feature);
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
