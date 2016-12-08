package util;

import java.util.PriorityQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.Instance;
import IO.DataManager;

public class HuffmanTree extends PrecomputedTree {
	private static Logger logger = LoggerFactory.getLogger(HuffmanTree.class);
	
	private static final long serialVersionUID = 6677270104977721765L;
	public static final String name = "HuffmanTree";
	protected int nLeaves;
	protected transient PriorityQueue<FreqTuple> freqheap;
	protected transient DataManager data;

	protected class FreqTuple implements Comparable<FreqTuple> {
		public float f;
		public TreeNode node;

		public FreqTuple(float f, TreeNode node) {
			this.f = f;
			this.node = node;
		}

		@Override
		public int compareTo(FreqTuple arg0) {
			if (f < arg0.f)
				return -1;
			if (f > arg0.f)
				return 1;
			return 0;
		}
	}

	public HuffmanTree(DataManager data, String treeFileName) {
		super(2, data.getNumberOfLabels());
		nLeaves = data.getNumberOfLabels();
		this.data = data;
		allocateFrequencies();
		buildHuffmanTree();
		writeTree(treeFileName);
	}

	public HuffmanTree(String treeFile) {
		super(treeFile);
	}

	protected void allocateFrequencies() {
		freqheap = new PriorityQueue<FreqTuple>(nLeaves);

		int[] counts = new int[nLeaves];
		int nInstances = 0;
		data.reset();
		logger.info( "Building Huffman tree...");
		while (data.hasNext() == true ) {
			Instance instance = data.getNextInstance();
			for (int i = 0; i < instance.y.length; i++) {
				counts[instance.y[i]]++;
			}
			nInstances++;
		}
		data.reset();
		
		logger.info( "Huffman tree is built based on " + nInstances + " instance." );
		
		TreeNode node;
		for (int j = 0; j < nLeaves; j++) {
			node = new TreeNode(j + 1);
			node.label = j;
			freqheap.add(new FreqTuple(((float) counts[j]) / nInstances, node));
			// Index is j+1 in order to store root of the tree at 0
			labelToIndex.put(j, j + 1);
			indexToNode.put(j + 1, node);
		}
	}

	public void buildHuffmanTree() {
		int currentIndex = nLeaves + 1; // Root is at 0
		TreeNode parent, c1, c2;
		for (int node = 0; node < this.numberOfInternalNodes; node++) {
			FreqTuple e1 = freqheap.poll();
			FreqTuple e2 = freqheap.poll();

			parent = new TreeNode(currentIndex);
			indexToNode.put(currentIndex, parent);
			c1 = e1.node;
			c2 = e2.node;
			c1.parent = parent;
			c2.parent = parent;
			parent.children.add(c1);
			parent.children.add(c2);

			FreqTuple tuple = new FreqTuple(e1.f + e2.f, parent);
			freqheap.add(tuple);
			currentIndex++;
		}
		FreqTuple root = freqheap.poll();
		indexToNode.put(0, root.node);
		indexToNode.remove(currentIndex - 1);
		root.node.index = 0;
		this.tree = root.node;
	}

}
