package Data;

import java.util.HashSet;

public class NodePCC implements Comparable<NodePCC> {

	protected HashSet<Integer> labelCombination;
	protected int label; 
	protected double p;

	public NodePCC(HashSet<Integer> labelCombination, int label, double p) {
		this.labelCombination = labelCombination;
		this.label = label;
		this.p = p;
	}

	public HashSet<Integer> getLabelCombination() {
		return this.labelCombination;
	}
	
	public int getLabel() {
		return this.label;
	}
	
	public double getP() {
		return this.p;
	}
	
	
	
	@Override
	public int compareTo(NodePCC o) {
		return Double.compare(-this.getP(),-o.getP());
	}
	
	
	@Override
	public String toString() {
		return new String("(" + this.labelCombination.toString() + ", " + this.label + ", " + this.p + ")");
	}
};


