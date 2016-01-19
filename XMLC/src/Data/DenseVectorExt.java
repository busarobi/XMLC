package Data;

import java.util.List;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class DenseVectorExt extends DenseVector {

	private static final long serialVersionUID = 1L;

	public DenseVectorExt(int length) {
		super( length );
	}
	
	public DenseVectorExt(double[] array) {
		super(array);		
	}
	
	public DenseVectorExt(List<Double> list) {
		super( list );
	}

    public DenseVectorExt(double[] array, int start, int end)
    {
    	super( array, start, end );
    }
    
    public DenseVectorExt(Vec toCopy)
    {
    	super( toCopy );
    }
	
	
    public double dot( AVPair[] av ){
    	double result = 0.0;
    	
    	for( int i = 0; i < av.length; i++ ){
    		result += (this.array[av[i].index] * av[i].value);
    	}
    	
    	return result;
    }
    
}
