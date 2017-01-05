package util;

import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.Serializable;
import java.io.Writer;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Scanner;

/**
 * Created by Kalina on 12.11.2016.
 */
public class PrecomputedTree extends Tree implements Serializable {
	private static final long serialVersionUID = 5656121729850759773L;
	private static Logger logger = LoggerFactory.getLogger(PrecomputedTree.class);
	final static public String name = "Precomputed";

	transient TreeNode tree;
	transient Map<Integer, TreeNode> indexToNode = new HashMap<Integer, TreeNode>();
	transient Map<Integer, Integer> labelToIndex = new HashMap<Integer, Integer>();

	public PrecomputedTree(int k, int m) {
		this.size = 2 * m - 1;
		this.k = k;
		this.numberOfInternalNodes = (int) this.size / 2;
	}

	public PrecomputedTree(String treeFileName) {
		initialize(treeFileName);
		this.size = this.indexToNode.size();
		this.numberOfInternalNodes = (int) this.size / 2;
	}

	public PrecomputedTree(ArrayList<Pair<Integer,Integer>> indices ) {
		for(int i = 0; i < indices.size(); i++ ){
			int parent = indices.get(i).getFirst();
			int child = indices.get(i).getSecond();

			if (parent == child) { // root
				this.tree = new TreeNode(parent);
				this.indexToNode.put(parent, this.tree);
			} else if (child > 0) { // internal node
				TreeNode node = new TreeNode(child);
				TreeNode parent_node = this.indexToNode.get(new Integer(parent));
				node.parent = parent_node;
				parent_node.children.add(node);
				this.indexToNode.put(child, node);
			} else { // leaf and label index
				int label = -child;
				TreeNode node = this.indexToNode.get(new Integer(parent));
				node.label = label;
				this.labelToIndex.put(label, parent);
				assert (node.children.size() == 0);
			}
		}
		
		super.initialize(this.k, this.m);
		this.size = this.indexToNode.size();
		this.numberOfInternalNodes = (int) this.size / 2;
	}
	
	
	public class TreeNode {
		public int index;
		public int label;
		public TreeNode parent;
		public List<TreeNode> children;

		public TreeNode(int index) {
			this.index = index;
			this.label = -1;
			this.parent = null;
			this.children = new ArrayList<TreeNode>();
		}

		public boolean isLeaf() {
			if (this.children.size() > 0)
				return false;
			return true;
		}
	}

	public void initialize(String treeFileName) {
		readTreeFile(treeFileName);
		super.initialize(this.k, this.m);
	}

	protected void readTreeFile(String treeFileName) {
		File file = null;
		Scanner inputScanner = null;
		try {

			file = new File(treeFileName);
			inputScanner = new Scanner(file);

			while (inputScanner.hasNextInt()) {
				int parent = inputScanner.nextInt();
				int child = inputScanner.nextInt();

				if (parent == child) { // root
					this.tree = new TreeNode(parent);
					this.indexToNode.put(parent, this.tree);
				} else if (child > 0) { // internal node
					TreeNode node = new TreeNode(child);
					TreeNode parent_node = this.indexToNode.get(new Integer(parent));
					node.parent = parent_node;
					parent_node.children.add(node);
					this.indexToNode.put(child, node);
				} else { // leaf and label index
					int label = -child;
					TreeNode node = this.indexToNode.get(new Integer(parent));
					node.label = label;
					this.labelToIndex.put(label, parent);
					assert (node.children.size() == 0);
				}
			}

			inputScanner.close();
		} catch (java.io.IOException e) {
			e.printStackTrace();
		}
	}

	protected String bfsTree() {
		StringBuilder str = new StringBuilder();

		LinkedList<Integer> indexes = new LinkedList<>();
		indexes.add(this.tree.index);
		str.append(this.tree.index + " " + this.tree.index + "\n");
		int i;
		ArrayList<Integer> children;
		while (!indexes.isEmpty()) {
			i = indexes.poll();
			if (isLeaf(i)) {
				str.append(i + " " + -getLabelIndex(i) + "\n");
			} else {
				children = getChildNodes(i);
				for (Integer child : children) {
					str.append(i + " " + child + "\n");
				}
				indexes.addAll(children);
			}
		}
		return str.toString();
	}

	protected void writeTree(String treeFileName) {
		File file;
		Writer wr;
		try {
			file = new File(treeFileName);
			wr = new FileWriter(file);
			wr.write(bfsTree());
			wr.close();
		} catch (java.io.IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public ArrayList<Integer> getChildNodes(int nodeIndex) {
		TreeNode node = this.indexToNode.get(new Integer(nodeIndex));
		if (!node.isLeaf()) {
			ArrayList<Integer> childNodes = new ArrayList<>();

			for (int i = 0; i < node.children.size(); i++) {
				childNodes.add(node.children.get(i).index);
			}
			return childNodes;
		} else {
			return null;
		}
	}

	@Override
	public int getParent(int nodeIndex) {
		TreeNode node = this.indexToNode.get(nodeIndex);
		if (node.parent == null)
			return -1;
		return node.parent.index;
	}

	public int getTreeIndex(int label) {
		return this.labelToIndex.get(label);
	}

	public int getLabelIndex(int nodeIndex) {
		return this.indexToNode.get(nodeIndex).label;
	}

	@Override
	public boolean isLeaf(int nodeIndex) {
		return this.indexToNode.get(nodeIndex).isLeaf();
	}

	public static void main(String[] argv) {
		String treeFile = "examples/huffmantree";
		System.out.println(treeFile);
		PrecomputedTree ct = new PrecomputedTree(treeFile);
		for (int i = 0; i < ct.indexToNode.size(); i++) {
			System.out.println("-------------------");
			TreeNode currNode = (TreeNode) ct.indexToNode.get(i);
			System.out.println("index: " + currNode.index);
			System.out.println("label: " + currNode.label);
			System.out.println("parent: " + ct.getParent(i));
			System.out.println("children: " + ct.getChildNodes(i));
			System.out.println("isLeaf: " + ct.isLeaf(i));
		}
		System.out.println("-------------------");
		int treeIndex = 2;
		System.out.println("Label of tree index " + treeIndex + ": " + ct.getLabelIndex(treeIndex));
		treeIndex = 5;
		System.out.println("Label of tree index " + treeIndex + ": " + ct.getLabelIndex(treeIndex));

		int labelIndex = 3;
		System.out.println("Tree index of label " + labelIndex + ": " + ct.getTreeIndex(labelIndex));
		labelIndex = 2;
		System.out.println("Tree index of label " + labelIndex + ": " + ct.getTreeIndex(labelIndex));

	}

}
