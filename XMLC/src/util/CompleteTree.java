package util;

import java.io.Serializable;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CompleteTree extends Tree implements Serializable {
	private static final long serialVersionUID = 5656121729850759773L;
	private static Logger logger = LoggerFactory.getLogger(CompleteTree.class);


	//int[] childNodes = new int[this.k];
	
	public CompleteTree(int k, int m) {
		initialize(k, m);
		
	}
	
	protected int computeSize(int k, int m) {
		
		double a = Math.pow(k, Math.floor(Math.log(m)/Math.log(k)));
		
		double b = m - a;
		
		double c = Math.ceil(b/(k-1.0));

		double d = (k*a - 1.0)/(k-1.0);

		double e = m - (a - c);
		
		return (int) e + (int) d;
	}
	
	
	@Override
	public void initialize(int k, int m) {
		super.initialize(k, m);
		this.size = computeSize(k,m);
		this.numberOfInternalNodes = this.size - m;
	}
	
	@Override
	public ArrayList<Integer> getChildNodes(int node) {
		
		if(node < this.numberOfInternalNodes) {
			
			//int[] childNodes = new int[this.k];
			ArrayList<Integer> childNodes = new ArrayList<Integer>(this.k); //new int[this.k];
			
			for(int i = 0; (i < k) && (k*node + i + 1 < this.size); i++) {
				//childNodes[i] = k  * node + i + 1;
				childNodes.add(k  * node + i + 1);
			}
			return childNodes;
		}
		else return null;
	}

	@Override
	public int getParent(int node) {
		if(node > 0) {
			return (int) Math.floor((node - 1.0) / (double) this.k);
		}
		else {
			return -1;
		}
	}

	public int getTreeIndex(int label) {
		return this.numberOfInternalNodes + label;
	}
	
	public int getLabelIndex(int treeIndex) {
		return treeIndex - this.numberOfInternalNodes;
	}
	
	@Override
	public boolean isLeaf(int node) {
		return (node >= this.numberOfInternalNodes);
	}
	
	public static void main(String[] argv) {

		int k = 2, m = 933;
		CompleteTree ct = new CompleteTree(k, m);
		
		logger.info(Integer.toString(ct.getSize()));
		
	}
	
}
