package Data;

public class EstimatePair implements Comparable<EstimatePair> {
    private double  p;
    private int label;

    public EstimatePair( int label, double p ){
    	this.label = label;
    	this.p = p;
    }
    
    public double getP() {
        return this.p;
    }

    public int getLabel() {
        return this.label;
    }
    
	public boolean equals(Object obj) {
        if(obj != null && obj instanceof EstimatePair) {
        	EstimatePair pair = (EstimatePair) obj;
            if(this.label != pair.label) return false;
            if(this.p != pair.p) return false;
            return true;
        	
        }
        return false;
    }
	
	public int hashCode() {
    
		int hash = 1;
        hash = hash * 17 + label;
        hash = hash * 31 + (int) p*1000;
        return hash;

	}

	
	@Override
	public int compareTo(EstimatePair o) {
		return Double.compare(-this.getP(),-o.getP());
	}
	
	
}