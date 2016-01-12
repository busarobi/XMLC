package util;

import java.util.ArrayList;

public abstract class Tree {

	protected int k = 2; //k-ary tree
	
	protected int m = 0; //number of leaves
	
	protected int size = 0; //size of the tree
	
	protected int numberOfInternalNodes = 0;
	
	public void initialize(int k, int m) {
		this.k = k ;
		this.m = m;
	}
	
	abstract public ArrayList<Integer> getChildNodes(int node);
	
	abstract public int getParent(int node);
	
	abstract public boolean isLeaf(int node); 
	
	abstract public int getTreeIndex(int label);
	
	abstract public int getLabelIndex(int treeIndex);
	
	public int getNumberOfLeaves() {
		return this.m;
	}
	
	public int getSize() {
		return this.size;
	}

	public int getNumberOfInternalNodes() {
		return this.numberOfInternalNodes;
	}
	
	
}
