package Data;


public class NodePLT {

	public int treeIndex;
	public double p;

	public NodePLT(int treeIndex, double p) {
		this.treeIndex = treeIndex;
		this.p = p;
	}

	@Override
	public String toString() {
		return new String("(" + this.treeIndex + ", " + this.p + ")");
	}
};


