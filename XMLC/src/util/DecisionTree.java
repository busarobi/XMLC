package util;

import java.util.ArrayList;

public abstract class DecisionTree {

	protected int k = 2; //binary tree
	
	protected int m = 0; //max. number of leaves
	
	protected int currentM = 0; //current number of leaves
	
	protected int size = 0; //size of the tree
	
	protected int numberOfInternalNodes = 0;
	
	DecisionTree(int m) {
		this.initialize(m);
	}
	
	public void initialize(int m) {
		
		this.m = m;
		this.size = 1;
	}
	
	abstract public boolean expandTree(int node);
	
	abstract public ArrayList<Integer> getChildNodes(int node);
	
	abstract public int getParent(int node);
	
	abstract public boolean isLeaf(int node); 
	
	public int getCurrentNumberOfLeaves() {
		return this.currentM;
	}
	
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
