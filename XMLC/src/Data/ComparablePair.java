package Data;

public class ComparablePair implements Comparable<ComparablePair> {
    private int  key;
    private double value;

    public ComparablePair( int key, double value ){
    	this.key = key;
    	this.value = value;
    }
    
    public int getKey() {
        return key;
    }

    public double getValue() {
        return value;
    }


	@Override
	public int compareTo(ComparablePair o) {
		return Double.compare(this.getKey(),o.getKey());
	}
}