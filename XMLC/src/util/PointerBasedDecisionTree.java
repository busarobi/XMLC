package util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;

import org.apache.logging.log4j.core.config.Node;

public class PointerBasedDecisionTree {

	protected DecisionTreeNode root = null;
	
	protected int maxSize = 0;
	protected int size = 0;
	
	int numOfSwaps = 0;
	
	public PointerBasedDecisionTree(int maxSize) {

		this.initialize(maxSize);
	}
	
	public void initialize(int maxSize) {
		
		this.root = new DecisionTreeNode(0);
		this.size = 0;
		this.maxSize = maxSize;
	}
	
	public DecisionTreeNode getRoot() {
		return this.root;
	}
	
	
	public boolean expandTree(DecisionTreeNode node) {
		
		if(node.isLeaf() && this.size < this.maxSize) {
			
			DecisionTreeNode left = new DecisionTreeNode(2*node.getNodeIndex()+1);
			DecisionTreeNode right = new DecisionTreeNode(2*node.getNodeIndex()+2);
			
			node.setLeftChild(left);
			node.setRightChild(right);
			//left.setParent(node);
			//right.setParent(node);
			this.size++;
			return true;
		} else {
			return false;
		}
	}
	
	public ArrayList<DecisionTreeNode> getChildNodes(DecisionTreeNode node) {
		
		if(!node.isLeaf()) {
			
			ArrayList<DecisionTreeNode> childNodes = new ArrayList<DecisionTreeNode>(2); //new int[this.k];
			childNodes.add(node.getLeftChild());
			childNodes.add(node.getRightChild());
			return childNodes;
		}
		else return null;
	}
	
	public DecisionTreeNode getParent(DecisionTreeNode node) {
		
		return node.getParent();
	}
	
	public boolean isLeaf(DecisionTreeNode node) {
		return node.isLeaf();
	}
	
	public int getMaximalSize() {
		return this.maxSize;
	}
	
	public int getSize() {
		return this.size;
	}
	
	public int getC() {
		return this.root.getC();
	}
	
	public int getNumOfSwaps() {
		return this.numOfSwaps;
	}
	
	public void swap(DecisionTreeNode node) {
		
		this.numOfSwaps++;
		
		DecisionTreeNode s = this.findLeafWithMinimalC();
		
		//System.out.println("Smallest leaf: " + s.getNodeIndex());
		
		DecisionTreeNode spa = s.getParent();
		DecisionTreeNode sgpa = spa.getParent();
		DecisionTreeNode ssib = spa.getRightChild().getNodeIndex() == s.getNodeIndex() ? spa.getLeftChild() : spa.getRightChild();
		
		if(spa.getNodeIndex() == sgpa.getLeftChild().getNodeIndex()) { 
			sgpa.setLeftChild(ssib);
		} else {
			sgpa.setRightChild(ssib);
		}
		
		this.updateC(ssib);
		this.size--;
		this.expandTree(node);
		
	}

	public DecisionTreeNode findLeafWithMinimalC() {
		
		DecisionTreeNode node = this.root;
		
		while(!node.isLeaf()) {
			//System.out.println("Search: " + node.getNodeIndex() + " " + node.getC() + " " +  node.getLeftChild().getNodeIndex() + " " + node.getLeftChild().getC() + " " +
			//			node.getRightChild().getNodeIndex() + " " + node.getRightChild().getC());
			if(node.getC() == node.getLeftChild().getC()) {
				node = node.getLeftChild();
			} else if (node.getC() == node.getRightChild().getC()) {
				node = node.getRightChild();
			}
			else {
				System.out.println("!!!");
				System.exit(-1); 
			}
		}
		
		
		return node; 
	}
	

	public void updateC(DecisionTreeNode node) {
		
		//System.out.println("Update: " + node.getNodeIndex() + " " + node.getC() + " " + this.root.getNodeIndex());
		
		while (node.getNodeIndex() != this.root.getNodeIndex() && node.getParent().getC() != node.getC()) {
			node = node.getParent();
			node.setC(Math.min(node.getRightChild().getC(), node.getLeftChild().getC()));
			//System.out.println("->" + node.getNodeIndex() + " " + node.getC() + " " +  node.getLeftChild().getNodeIndex() + " " + node.getLeftChild().getC() + " " +
			//		node.getRightChild().getNodeIndex() + " " + node.getRightChild().getC());
		}
		
	}
	
	public int height() {
		DecisionTreeNode node = this.root;
		return node.height();
	}
 	
	
	public String toString() {
	
		DecisionTreeNode node = this.root;
		return node.toNewickString();
		
	}
}
