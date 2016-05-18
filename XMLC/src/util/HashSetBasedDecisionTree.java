package util;

import java.util.ArrayList;
import java.util.HashSet;

public class HashSetBasedDecisionTree extends DecisionTree{

	protected HashSet<Integer> leaves = null;
	
	public HashSetBasedDecisionTree(int m) {
		super(m);
		this.initialize(m);
	}
	
	public void initialize(int m) {
		super.initialize(m);
		this.leaves = new HashSet<>(this.m);
		this.leaves.add(0);
	}
	
	public boolean expandTree(int node) {
		
		if(this.isLeaf(node) && this.currentM < this.m) {
			this.leaves.remove(node);
			this.leaves.add(2*node+1);
			this.leaves.add(2*node+2);
			this.currentM++;
			this.size++;
			return true;
		} else {
			return false;
		}
	}
	
	public ArrayList<Integer> getChildNodes(int node) {
		
		if(!this.isLeaf(node)) {
			
			ArrayList<Integer> childNodes = new ArrayList<Integer>(2); //new int[this.k];
			childNodes.add(2*node+1);
			childNodes.add(2*node+2);
			return childNodes;
		}
		else return null;
	}
	
	public int getParent(int node) {
		if(node > 0) {
			return (int) Math.floor((node - 1.0) / (double) 2);
		}
		else {
			return -1;
		}
	}
	
	public boolean isLeaf(int node) {
		return this.leaves.contains(node);
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
