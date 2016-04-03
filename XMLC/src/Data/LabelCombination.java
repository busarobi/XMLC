package Data;

import java.util.HashSet;

public class LabelCombination implements Comparable<LabelCombination> {
    private HashSet<Integer> labelCombination = null;
    private double p;

    public LabelCombination(HashSet<Integer> labelCombination, double p ){
    	this.labelCombination = labelCombination;
    	this.p = p;
    }
    
    public double getP() {
        return this.p;
    }

    public HashSet<Integer> getLabelCombination() {
        return this.labelCombination;
    }
    
	public boolean equals(Object obj) {
        if(obj != null && obj instanceof LabelCombination) {
        	LabelCombination lc = (LabelCombination) obj;
            if(!this.labelCombination.equals(lc)) return false;
            if(this.p != lc.p) return false;
            return true;
        	
        }
        return false;
    }
	
	public int hashCode() {
    
		int hash = 1;
        hash = hash * 17 + this.labelCombination.hashCode();
        hash = hash * 31 + (int) p*1000;
        return hash;

	}

	
	@Override
	public int compareTo(LabelCombination o) {
		return Double.compare(-this.getP(),-o.getP());
	}
	
	public String toString() {
		return "[" + this.labelCombination.toString() + ", " + this.p + "]";
	}
	
}