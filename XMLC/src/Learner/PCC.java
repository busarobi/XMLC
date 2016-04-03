package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.LabelCombination;
import Data.NodePCC;
import Learner.step.StepFunction;
import preprocessing.FeatureHasherFactory;

public class PCC extends MLLRFH {

	private static final long serialVersionUID = 1468223396740995671L;

	private static Logger logger = LoggerFactory.getLogger(PCC.class);

	protected double epsilon = 0.0;
	
	transient protected int[] Tarray = null;	
	protected double[] scalararray = null;
	
	
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}
	
	public double getEpsilon() {
		return this.epsilon;
	}
	
	public PCC(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Learner: PCC" );
		logger.info("#####################################################" );
		
		// epsilon
		this.epsilon = Double.parseDouble(this.properties.getProperty("epsilon", "0.0"));
		logger.info("#### epsilon: " + this.epsilon);

	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

		logger.info( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info("#####################################################" );
			
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.m);
		
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.bias = new double[this.m];
		
		this.Tarray = new int[this.m];
		this.scalararray = new double[this.m];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
		
		logger.info( "Done." );
	}
	
		
	@Override
	public void train(AVTable data) {
				
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1), this.epochs );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {

				int currIdx = indirectIdx.get(i);
				
				HashSet<Integer> positiveLabels = new HashSet<Integer>();
				for (int j = 0; j < traindata.y[currIdx].length; j++) {
					positiveLabels.add(traindata.y[currIdx][j]);
				}
					
				for(int j = m - 1; j >= 0; j--) {
				
					double y = positiveLabels.remove(j)? 1.0 : 0.0;
					double posterior = getPartialPosteriors(traindata.x[currIdx], positiveLabels, j);
					double inc = -(y - posterior); 
					updatedPosteriors(currIdx, positiveLabels, j, inc);
				}

				this.T++;

				if ((i % 100000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					//logger.info("Weight: " + this.w[0].get(0) );
					logger.info("Scalar: " + this.scalar);
				}
			}

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
		}
		
		int zeroW = 0;
		double sumW = 0;
		int maxNonZero = 0;
		int index = 0;
		for(double weight : w) {
			if(weight == 0) zeroW++;
			else maxNonZero = index;
			sumW += weight;
			index++;
		}
		logger.info("Hash weights (lenght, zeros, nonzeros, ratio, sumW, last nonzero): " + w.length + ", " + zeroW + ", " + (w.length - zeroW) + ", " + (double) (w.length - zeroW)/(double) w.length + ", " + sumW + ", " + maxNonZero);
	}


	protected void updatedPosteriors( int currIdx,  HashSet<Integer> labelFeatures, int label, double inc) {
			
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		int n = traindata.x[currIdx].length;
		
		for(int i = 0; i < n; i++) {

			int index = fh.getIndex(label, traindata.x[currIdx][i].index);
			int sign = fh.getSign(label, traindata.x[currIdx][i].index);
			
			double gradient = this.scalararray[label] * inc * (traindata.x[currIdx][i].value * sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[index] -= update; 
			
		}
		
		for(int l : labelFeatures) {
			
			int hi = fh.getIndex(label,  this.d + l); 
			int sign = fh.getSign(label, this.d + l);
			
			double gradient = this.scalararray[label] * inc * (sign);
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[hi] -= update;
			
		}
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
	}
	
	public double getFeatureScore(AVPair[] x, int label) {
		
		double score = 0.0;
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			score += (x[i].value *sign) * (1/this.scalararray[label]) * this.w[hi];
		}
		
		score += (1/this.scalararray[label]) * this.bias[label];
		
		return score;
	}
	
	public double getLabelScore( HashSet<Integer> labelFeatures, int label) {
		
		double score = 0.0;
		
		for(int l : labelFeatures) {
			
			int hi = fh.getIndex(label,  this.d + l); 
			int sign = fh.getSign(label, this.d + l);
			score += (sign) * (1/this.scalararray[label]) * this.w[hi];
		}
		
		return score;
		
	}
	
	
	@Override
	public double estimateLabelCombinationProbability(AVPair[] x, int[] y) {
		
		double estimate = 1;
		HashSet<Integer> positiveLabels = new HashSet<Integer>();
		for (int j = 0; j < y.length; j++) {
			positiveLabels.add(y[j]);
		}
		HashSet<Integer> labelFeatures = new HashSet<Integer>();
		for(int j = 0; j < m; j++) {
			double z = positiveLabels.contains(j) ? 1.0 : 0.0;
			double p = this.getPartialPosteriors(x,labelFeatures, j); 
			estimate *= (z == 1.0?  p : 1 - p);
			if(z == 1.0) labelFeatures.add(j);
		}
		return estimate;
	}

	
	public double getPartialPosteriors(AVPair[] x, HashSet<Integer> labelFeatures, int label) {
		
		double posterior = getFeatureScore(x, label);
		
		posterior += getLabelScore(labelFeatures, label);
		
		//posterior += (1/this.scalararray[label]) * this.bias[label]; 
		
		posterior = s.value(posterior);		
		
		return posterior;
	}
	
	@Override
	public TreeSet<LabelCombination> getTopKLabelCombinations(AVPair[] x, int k) {

		double[] scores = new double[this.m];
		
		for(int j = 0; j < m; j++) {
			scores[j] = getFeatureScore(x,j);
		}
		
		TreeSet<LabelCombination> positiveLabelCombinations = new TreeSet<LabelCombination>();

	    int foundTop = 0;
	    
	    //NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePCC> queue = new PriorityQueue<NodePCC>(this.m, null);
		PriorityQueue<NodePCC> heirless = new PriorityQueue<NodePCC>(this.m, null);
		
		double eps = this.epsilon;
		
		queue.add(new NodePCC(new HashSet<Integer>(), 0, 1.0));
		
		while(!queue.isEmpty() && (foundTop < k)) {

			NodePCC node = queue.poll();

			int label = node.getLabel();
			
			if(label < m) {
			
				AbstractLearner.numberOfInnerProducts++;
				
				double f = scores[label] + this.getLabelScore(node.getLabelCombination(), label); 
				
				double currentP = s.value(f); 
				
				
				boolean survived = false;
			
				double p = node.getP() * (currentP);

				if(p >= eps) {
					
					HashSet<Integer> child = new HashSet<Integer>(node.getLabelCombination());
					child.add(label);
					
					queue.add(new NodePCC(child, label + 1, p));
					survived = true;
				}
				
				p = node.getP() * (1 - currentP);
				
				if(p >= eps) {
					
					HashSet<Integer> child = new HashSet<Integer>(node.getLabelCombination());
					
					queue.add(new NodePCC(child, label + 1, p));
					
					survived = true;
				}
				if(survived == false) {
					
					HashSet<Integer> child = new HashSet<Integer>(node.getLabelCombination());
					if(currentP > 0.5) child.add(label);
					
					heirless.add(new NodePCC(child, label + 1, node.getP() * Math.max(1 - currentP,currentP)));
				}
				
			} else {
				positiveLabelCombinations.add(new LabelCombination(node.getLabelCombination(), node.getP()));
				foundTop++;
			}
		}
		
		eps = 0.0;
		
		while(!heirless.isEmpty() && (foundTop < k)) {

			NodePCC node = heirless.poll();
			int label = node.getLabel();
			double  p = node.getP();
			
			HashSet<Integer> labelCombination = node.getLabelCombination();
			
			while(label < this.m) {
				
				AbstractLearner.numberOfInnerProducts++;
				
				double f = scores[label] + this.getLabelScore(node.getLabelCombination(), label); 
				
				double currentP = s.value(f); 
				
				
				if(currentP < 0.5) {
					p *= (1-currentP);
					if(p < eps) break;
					label++;
					
				} else {
					p *= (currentP);
					if(p < eps) break;
					labelCombination.add(label);
					label++;
					
				}
			}
		
			if(label == this.m) {
				positiveLabelCombinations.add(new LabelCombination(labelCombination, p));
				foundTop++;
				eps = positiveLabelCombinations.first().getP();
			}
		}
		return positiveLabelCombinations;
	}

	
	public void save(String fname) {
		}

	public void load(String fname) {
	}
	
	
	
}
