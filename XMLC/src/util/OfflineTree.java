package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.sourceforge.olduvai.treejuxtaposer.TreeParser;
import net.sourceforge.olduvai.treejuxtaposer.drawer.TreeNode;

public class OfflineTree extends Tree implements Serializable {
	private static final long serialVersionUID = 5656121729850759773L;
	private static Logger logger = LoggerFactory.getLogger(OfflineTree.class);
	
	net.sourceforge.olduvai.treejuxtaposer.drawer.Tree newickTree = null;
	
	public OfflineTree(int k, int m) {
		logger.info("Invalid NewickOfflineTree constructor!");
		System.exit(-1);		
	}
	
	public OfflineTree(String treeFileName) {
		initialize(treeFileName);
		this.size = this.newickTree.nodes.size();
		this.numberOfInternalNodes = (int)this.newickTree.nodes.size()/2;
	}

	public void initialize(String treeFileName) {
		readTreeFile(treeFileName);
		super.initialize(this.k, this.m);
	}
	
	private void readTreeFile(String treeFileName){
		FileInputStream fstream = null;
		BufferedReader br = null;
		try {
			fstream = new FileInputStream(treeFileName);	
			br = new BufferedReader(new InputStreamReader(fstream));
			TreeParser tp = new TreeParser(br);
	        newickTree = tp.tokenize(1, treeFileName, null);	        
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}      
	}

	@Override
	public ArrayList<Integer> getChildNodes(int node) {		
		TreeNode currNode = this.newickTree.getNodeByKey(node);
		if(currNode.isLeaf()){
			return null;
		}else{
			ArrayList<Integer> childNodes = new ArrayList<Integer>(this.k);
			for(int i = 0; i<currNode.numberChildren(); i++){
				childNodes.add(currNode.getChild(i).key);
			}
			return childNodes;
		}
	}

	@Override
	public int getParent(int node) {
		TreeNode currNode = this.newickTree.getNodeByKey(node);
		if(currNode.isRoot()){
			return -1;
		}else{
			return currNode.parent.key;
		}
	}

	public int getTreeIndex(int label) {
		TreeNode currNode = this.newickTree.getNodeByName(Integer.toString(label)); 
		return currNode.key;
	}
	
	public int getLabelIndex(int treeIndex) {
		return Integer.parseInt(this.newickTree.getNodeByKey(treeIndex).getName());			
	}
	
	public int getNodeDepth(int node){
		TreeNode currNode = (TreeNode) this.newickTree.nodes.get(node); 
		return currNode.height;
	}
	
	public boolean isBalanced() {
		ArrayList<Integer> depths = new ArrayList<Integer>();
		
		for(int i = 0; i<this.newickTree.nodes.size(); i++){
			TreeNode currNode = (TreeNode) this.newickTree.nodes.get(i); 
			if(currNode.isLeaf()){
				depths.add(currNode.height);
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
		return this.newickTree.getNodeByKey(node).isLeaf();		
	}
	
	public static void main(String[] argv) {
		//String treeFile = "/windows/projects/xmlc/XMLC/trees/newick2";
		String treeFile = "/windows/projects/xmlc/data/test/trees/test_newick.py";
		System.out.println(treeFile);
		OfflineTree ct = new OfflineTree(treeFile);
		for(int i = 0; i<ct.newickTree.nodes.size(); i++){
			System.out.println("-------------------");
			TreeNode currNode = (TreeNode) ct.newickTree.nodes.get(i); 
			//	return name + "(" + key + " @ " + height + ")";
			System.out.println("name: " + currNode.getName());
			System.out.println("key: " + currNode.key);
			System.out.println("height: " + currNode.height);
			System.out.println("label: " + currNode.label);
			System.out.println("weight: " + currNode.weight);
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
		labelIndex = 5;
		System.out.println("Tree index of label " + labelIndex + ": " + ct.getTreeIndex(labelIndex));
				
	}
	
}
