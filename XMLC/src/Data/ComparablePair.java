package Data;


public class ComparablePair implements Comparable<ComparablePair> {
    private double  key;
    private int value;

    public ComparablePair( double key, int value ){
    	this.key = key;
    	this.value = value;
    }
    
    public double getKey() {
        return key;
    }

    public int getValue() {
        return value;
    }


	@Override
	public int compareTo(ComparablePair o) {
		return Double.compare(this.getKey(),o.getKey());
	}
}	
