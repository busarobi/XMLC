package threshold;


public class ComparablePairEUM implements Comparable<ComparablePairEUM> {
    private double  key;
    private int value;

    public ComparablePairEUM( double key, int value ){
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
	public int compareTo(ComparablePairEUM o) {
		return Double.compare(this.getKey(),o.getKey());
	}
}	
