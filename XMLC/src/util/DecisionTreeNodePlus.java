package util;

import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

public class DecisionTreeNodePlus {
	
	int nodeIndex;

	boolean isLeaf = true;
	
	DecisionTreeNodePlus parent = null;
	DecisionTreeNodePlus left = null;
	DecisionTreeNodePlus right = null;
	
	double bias = 0.0;
	int T = 1;
	double scalar = 1.0;
	
	HashMap<Integer,Double> sumOfScoresForClass = new HashMap<>();
	HashMap<Integer,Integer> classDistribution = new HashMap<>();
	HashMap<Integer,Integer> numberOfExamplesUsedToTrainFromEachClass = new HashMap<>();
	HashMap<Integer,Double> expectedScoreForClass = new HashMap<>();
	double expectedScoreNominator = 0.0;
	double expectedScoreDenominator = 0.0;
	double expectedScore = 0.0;
	int denominator = 0;
	int C = 0;
	
	Random random = new Random();

	public DecisionTreeNodePlus(int nodehash) {
		this.nodeIndex = nodehash;
	}
	
	public int getNodeIndex() {
		return this.nodeIndex;
	}
	
	public DecisionTreeNodePlus getParent() {
		return this.parent;
	}
	
	public DecisionTreeNodePlus getLeftChild() {
		return this.left;
	}
	
	public DecisionTreeNodePlus getRightChild() {
		return this.right;
	}
	
	public void setParent(DecisionTreeNodePlus parent) {
		this.parent = parent;
	}
	
	public void setLeftChild(DecisionTreeNodePlus left) {
		this.isLeaf = false;
		this.left = left;
		this.left.setParent(this);
	}
	
	public void setRightChild(DecisionTreeNodePlus right) {
		this.isLeaf = false;
		this.right = right;
		this.right.setParent(this);
	}
	
	public void updateNode(int label) {
		
		if(!this.classDistribution.containsKey(label)) {
			double rv1 = 0.05*(2*random.nextDouble()-1);
			this.classDistribution.put(label, 0);
			this.sumOfScoresForClass.put(label, rv1);
			this.numberOfExamplesUsedToTrainFromEachClass.put(label, 0);
			this.expectedScoreForClass.put(label, rv1);
			this.expectedScoreNominator = 0;
		}
		this.classDistribution.put(label, this.classDistribution.get(label) + 1);
		this.denominator++;
	}
	
	public boolean isLeaf() {
		return this.isLeaf;
	}
	
	
	public void updateStatistics(int label, double posterior) {
		
		
		this.expectedScoreDenominator += 1;
		
		double step = 1 /Math.sqrt(this.expectedScoreDenominator);
		
		this.numberOfExamplesUsedToTrainFromEachClass.put(label, this.numberOfExamplesUsedToTrainFromEachClass.get(label) + 1);
		this.sumOfScoresForClass.put(label, this.sumOfScoresForClass.get(label) + step * posterior);
		this.expectedScoreForClass.put(label, this.sumOfScoresForClass.get(label));// / (double) this.numberOfExamplesUsedToTrainFromEachClass.get(label));
		this.expectedScoreNominator += step*posterior;
		
		this.expectedScore = this.expectedScoreNominator / this.expectedScoreDenominator;
		
	}

	public double getExpectedScore() {
		return this.expectedScore;
	}
	
	public double getExpectedScore(int label) {
		return this.expectedScoreForClass.get(label);
	}
	
	public int getNumberOfClasses() {
		return this.classDistribution.size();
	}
	
	public int getC() {
		return this.C;
	}
	
	public void setC(int C) {
		this.C = C;
	}
	
	public int getT() {
		return this.T;
	}
	
	public double getScalar() {
		return this.scalar;
	}
	
	public double getBias() {
		return this.bias;
	}
	
	public void setT(int T) {
		this.T = T;
	}
	
	public void setScalar(double scalar) {
		this.scalar = scalar;
	}
	
	public void setBias(double bias) {
		this.bias = bias;
	}
	
	public double getProbability(int label) {
		return this.classDistribution.get(label) / (double) this.denominator;
	}
	
	public Set<Integer> getLabels() { 
		return this.classDistribution.keySet();
	}
	
	public HashMap<Integer,Integer> getClassDistribution() {
		return this.classDistribution;
	}
	
	public int getMaxFrequency() {
		//System.out.println(this.classDistribution.toString());
		//System.out.println(Collections.max(this.classDistribution.values()));
		return Collections.max(this.classDistribution.values());
	}
	
	public int height() {
		
		int height = 0; 
		if(!this.isLeaf()) {
			int leftHeight = this.getLeftChild().height();
			int rightHeight = this.getRightChild().height();
			height =  Math.max(leftHeight, rightHeight) + 1;
		}
		return height;
		
	}

	public String toNewickString() {
		
		String newick = ""; 
		if(this.isLeaf()) {
			newick += this.getNodeIndex();
		} 
		else {
			newick = this.getNodeIndex() + newick;
			newick = ")" + newick; 
			newick = this.getLeftChild().toNewickString() + newick;
			newick = "," + newick;
			newick = this.getRightChild().toNewickString() + newick;
			newick = "(" + newick;
		}
		return newick;
		
	}

}
