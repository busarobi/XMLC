package Data;

public class ComparableTriplet implements Comparable<ComparableTriplet> {
    private double key;
    private int value;
    private int y;

    public ComparableTriplet( double key, int value, int y ){
    	this.key = key;
    	this.value = value;
    	this.y = y;
    }
    
    public double getKey() {
        return key;
    }

    public int getValue() {
        return value;
    }
    
    public int gety() {
    	return y;
    }


	@Override
	public int compareTo(ComparableTriplet o) {
		return Double.compare(-this.getKey(),-o.getKey());
	}}