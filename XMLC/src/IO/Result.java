package IO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;


public class Result {
	private static Logger logger = LoggerFactory.getLogger(Result.class);

	protected double[][] posteriors = null;
	protected double HL = 0.0;
	

	public Result( double[][] posteriors, int[][] y, int n, int m ) {
		init( posteriors, y, n, m );
	}
	
	public Result( double[][] posteriors, AVTable data ) {
		init( posteriors, data.y, data.n, data.m );
	}
	
	private void init(double[][] posteriors, int[][] y, int n, int m) {
		this.posteriors = posteriors;		
		
		
		for(int i = 0; i < posteriors.length; i++ ) {
			int indexy = 0;
			for(int j = 0; j < posteriors[i].length; j++ ) {
				int forecast =  (posteriors[i][j] > 0.5)? 1:0;
				
				int tl;
				if (y[i] == null ) tl = 0;
				else tl = y[i].length;
				
				if (indexy < tl) {
				    if ( (y[i][indexy]!=j) && (forecast == 1 ) )  {
						this.HL += 1.0;
					} 
				    if ( (y[i][indexy]==j) && (forecast == 0 ) )  {
						this.HL += 1.0;
					}
					if ( y[i][indexy] == j ) { 
						indexy++;				
					}
				
			    } else if ( forecast == 1 ) {
			    	this.HL += 1.0;
			    }		
			}
		}
		
		this.HL = this.HL / ((double)(n * m)); 

		
	}

	public double getHL() {
		return HL;
	}

	public static void main(String[] args) throws Exception {
		//String fileName = "/Users/busarobi/work/XMLC/data/scene/scene_train";
		String trainfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_train.svm";
		String testfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_train.svm";
		
		
		DataReader testdatareader = new DataReader( testfileName );
		AVTable testdata = testdatareader.read();
		
		double[][] posteriors = new double[testdata.n][];
		for( int i = 0; i < testdata.n; i++ ) {
			posteriors[i] = new double[testdata.m];
			for(int j=0; j < testdata.y[i].length; j++ ){
				posteriors[i][testdata.y[i][j]] = 1.0;
			}
		}
	
		Result r2 = new Result(posteriors, testdata);
		//Result r2 = new Result(testdata.y, testdata);
		logger.info( "Hamming loss: " + r2.getHL());

		
	}
}
