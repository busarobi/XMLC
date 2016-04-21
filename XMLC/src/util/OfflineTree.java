package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OfflineTree extends Tree implements Serializable {
	private static final long serialVersionUID = 5656121729850759773L;
	private static Logger logger = LoggerFactory.getLogger(OfflineTree.class);
	
	public static final int INTERNAL = 1;
	public static final int LEAF = 2;
	public static final int NONE = 0;
	
	private ArrayList<Integer> labels;
	private ArrayList<Integer> tree;
	private HashMap<Integer, Integer> leafIndexToLabel; 
	private HashMap<Integer, Integer> labelToLeafIndex; 
	
	public OfflineTree(int k, int m) {
		logger.info("Invalid OfflineTree constructor!");
		System.exit(-1);		
	}
	
	public OfflineTree(String treeFileName) {
		initialize(treeFileName);			
	}

	public void initialize(String treeFileName) {
		readTreeFile(treeFileName);
		super.initialize(this.k, this.m);
	}
	
	private void readTreeFile(String treeFileName){
		FileInputStream fstream = null;
		BufferedReader br = null;
		String line;
		this.labels = new ArrayList<Integer>();
		this.tree = new ArrayList<Integer>();
		this.leafIndexToLabel = new HashMap<Integer, Integer>(); 
		this.labelToLeafIndex = new HashMap<Integer, Integer>(); 
		
		int numLabels = 0, treeSize = 0, branchingFactor = 0;
		try {
			fstream = new FileInputStream(treeFileName);
			br = new BufferedReader(new InputStreamReader(fstream));
			numLabels = Integer.parseInt(br.readLine());
			treeSize = Integer.parseInt(br.readLine());
			branchingFactor = Integer.parseInt(br.readLine());
			
			for(int i = 0; i<numLabels; i++ ){
				if((line = br.readLine()) != null){
					this.labels.add(Integer.parseInt(line));
				}
			}
			for(int i = 0; i<treeSize; i++ ){
				if((line = br.readLine()) != null){
					if(line.equals("internal")){
						this.tree.add(INTERNAL);
					}else if(line.equals("none")){
						this.tree.add(NONE);
					}else{
						this.tree.add(LEAF);
						this.leafIndexToLabel.put(i,Integer.parseInt(line));
						this.labelToLeafIndex.put(Integer.parseInt(line), i);
					}
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		} finally{
			if (br != null){
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		this.k = branchingFactor;
		this.m = numLabels;
		this.size = treeSize;
	}
	
	@Override
	public ArrayList<Integer> getChildNodes(int node) {		
		if(this.tree.get(node) == INTERNAL){
			ArrayList<Integer> childNodes = new ArrayList<Integer>(this.k);
			for(int i = 1; i <= k; i++) {
				int childIdx = k  * node + i;
				if(this.tree.get(node) != NONE){
					childNodes.add(childIdx);
				}
			}
			return childNodes;
		} else {
			return null;
		}
	}

	@Override
	public int getParent(int node) {
		if(node > 0) {
			return (int) Math.floor((node - 1.0) / (double) this.k);
		}
		return -1;		
	}

	public int getTreeIndex(int label) {
		return this.labelToLeafIndex.get(label);		
	}
	
	public int getLabelIndex(int treeIndex) {
		return this.leafIndexToLabel.get(treeIndex);			
	}
	
	public int getNodeDepth(int node){
		int depth = 0;
		int idx = node;
		while(idx != 0){
			idx = getParent(idx);
			depth += 1;
		}
		return depth;
	}
	
	public boolean isBalanced() {
		ArrayList<Integer> depths = new ArrayList<Integer>();
		for(int i = 0; i < this.size; i++){
			if(this.tree.get(i) == LEAF){
				depths.add(getNodeDepth(i));
			}
		}	
		HashMap<Integer, Integer> frequencymap = new HashMap<Integer, Integer>();
		for(Integer d : depths) {
		  if(frequencymap.containsKey(d)) {
		    frequencymap.put(d, frequencymap.get(d)+1);
		  }
		  else{ frequencymap.put(d, 1); }
		}
		if (frequencymap.keySet().size() <= 2)
			return true;
		return false;
	}	
	@Override
	public boolean isLeaf(int node) {
		if(this.tree.get(node) == LEAF)
			return true;
		return false;
	}
	
	public static void main(String[] argv) {
		String treeFile = "/windows/projects/xmlc/data/wiki10/tree/wiki10_train-processed.txt";
		System.out.println(treeFile);
		OfflineTree ct = new OfflineTree(treeFile);
		System.out.println(Integer.toString(ct.getSize()));
		System.out.println(ct.getParent(100));
		System.out.println(ct.getChildNodes(100));
		System.out.println(ct.getTreeIndex(20751));
		System.out.println(ct.isLeaf(10));
		System.out.println(ct.isLeaf(65534));
		System.out.println(ct.isBalanced());
		
	}
	
}
