package util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.PriorityQueue;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.Instance;
import IO.DataManager;

public class HuffmanTree extends Tree implements Serializable {
	protected class FreqTuple implements Comparable<FreqTuple> {
		public int index;
		public float f;

		public FreqTuple(int i, float f) {
			index = i;
			this.f = f;
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

	private static final long serialVersionUID = 5425467548503925653L;

	private static Logger logger = LoggerFactory.getLogger(HuffmanTree.class);
	public static final String name = "Huffman";
	protected int nLeaves;
	protected long[] nodes;
	protected int[] parent;
	protected transient DataManager data;

	protected transient PriorityQueue<FreqTuple> freqheap;

	public HuffmanTree(DataManager data) {
		logger.info("Initializing Huffman tree...");
		this.k = 2;
		nLeaves = data.getNumberOfLabels();
		logger.info("Number of tree leaves: " + nLeaves);
		size = 2 * nLeaves - 1;
		numberOfInternalNodes = nLeaves - 1;
		nodes = new long[size];
		parent = new int[size];
		for (int i = 0; i < nLeaves; i++) {
			nodes[i] = i;
		}
		this.data = data;
		allocateFrequencies();
		super.initialize(k, nLeaves);
		this.buildHuffmanTree();
	}

	protected void allocateFrequencies() {
		freqheap = new PriorityQueue<FreqTuple>(nLeaves);

		int[] counts = new int[nLeaves];
		int nInstances = 0;
		data.reset();
		while (data.hasNext()) {
			Instance instance = data.getNextInstance();
			for (int i = 0; i < instance.y.length; i++) {
				counts[instance.y[i]]++;
			}
			nInstances++;
		}
		data.reset();
		for (int j = 0; j < nLeaves; j++) {
			freqheap.add(new FreqTuple(j, ((float) counts[j]) / nInstances));
		}
	}

	public void buildHuffmanTree() {
		long code;
		int currentIndex = nLeaves;
		for (int node = 0; node < this.numberOfInternalNodes; node++) {
			FreqTuple e1 = freqheap.poll();
			FreqTuple e2 = freqheap.poll();

			code = childrenToCode(e1.index, e2.index);
			parent[e1.index] = currentIndex;
			parent[e2.index] = currentIndex;
			nodes[currentIndex] = code;

			FreqTuple newNode = new FreqTuple(currentIndex, e1.f + e2.f);
			freqheap.add(newNode);
			currentIndex++;
		}
	}

	protected long childrenToCode(int c1, int c2) {
		int lo = Math.min(c1, c2);
		int hi = Math.max(c1, c2);
		return lo * (size - 1) - lo * (lo - 1) / 2 + hi;
	}

	protected ArrayList<Integer> codeToChildren(long code) {
		int i = Math.floorDiv(size, 2);
		int lo = 0;
		int hi = size;
		// Use bisection to find the first child:
		int ind = divideCode(code, i);
		int indm1 = divideCode(code, i - 1);
		while (!(ind == 0 && indm1 > 0)) {
			if (ind < 1) {
				hi = i;
				i -= Math.ceil((i - lo + 1) / 2.0) - 1;
			} else {
				lo = i;
				i += (hi - i + 1) / 2;
			}
			if (i == 1)
				break;
			ind = divideCode(code, i);
			indm1 = divideCode(code, i - 1);
		}
		i -= 1;
		ArrayList<Integer> result = new ArrayList<Integer>(2);
		result.add(i);
		result.add((int) (code - i * size + i * (i - 1) / 2 + i));
		return result;
	}

	protected int divideCode(long code, int index) {
		return (int) (code / (index * size - index * (index - 1) / 2));
	}

	/**
	 * Given a code compute the indeces of the children. The first index is
	 * determined using bisection. Complexity: O(log n) where n is the size of
	 * the tree.
	 * If the node is a leaf, return an empty ArrayList.
	 * 
	 * @param index of the node
	 * @return ArrayList containing the indeces of the children with i1 < i2
	 *         or empty, if the node is a leaf node
	 */
	@Override
	public ArrayList<Integer> getChildNodes(int node) {
		if (isLeaf(node)) {
			return new ArrayList<Integer>(0);
		}
		long code = nodes[node];
		return codeToChildren(code);
	}

	@Override
	public int getLabelIndex(int treeIndex) {
		return treeIndex;
	}

	@Override
	public int getParent(int node) {
		return parent[node];
	}

	@Override
	public int getTreeIndex(int label) {
		return label;
	}

	@Override
	public boolean isLeaf(int node) {
		return (node <= this.nLeaves);
	}

}
