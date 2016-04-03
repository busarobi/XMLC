package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Properties;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.EstimatePair;
import Learner.step.StepFunction;
import preprocessing.FeatureHasherFactory;
import util.CompleteTree;

public class FT extends MLLRFH {

	private static final long serialVersionUID = 1468223396740995671L;

	private static Logger logger = LoggerFactory.getLogger(FT.class);

	transient protected int t = 0;
	
	transient CompleteTree tree = null;
	
	transient protected int[] Tarray = null;	
	protected double[] scalararray = null;
	
	public FT(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);

		logger.info("#####################################################" );
		logger.info("#### Learner: FT" );
		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		
		
		this.tree = new CompleteTree(2, this.m);
		
		this.t = this.tree.getSize(); 

		logger.info( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info( "#### Num. of inner node of the trees: " + this.t  );
		logger.info("#####################################################" );
			
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, this.t);
		
		logger.info( "Allocate the learners..." );

		this.w = new double[this.hd];
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}
		
		logger.info( "Done." );
		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);
		
		//logger.info( "Done." );
	}
	
		
	@Override
	public void train(AVTable data) {
				
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1), this.epochs );
			// random permutation
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			
			for (int i = 0; i < traindata.n; i++) {

				int currIdx = indirectIdx.get(i);

				int treeIndex = this.tree.getTreeIndex(traindata.y[currIdx][0]); 
				int node = treeIndex;
				treeIndex = (int) this.tree.getParent(treeIndex); 
				
				while(treeIndex >= 0) {

					double posterior = getPartialPosteriors(traindata.x[currIdx],treeIndex);
					double inc = -((node % 2) - posterior); 

					updatedPosteriors(currIdx, treeIndex, inc);
					
					if(Math.abs(inc) >= 0.5) break;
					
					node = treeIndex;
					treeIndex = (int) this.tree.getParent(treeIndex); 
				
				}
				
				//logger.info("Negative tree indices: " + negativeTreeIndices.toString());

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


	protected void updatedPosteriors( int currIdx, int label, double inc) {
			
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
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;
		//logger.info("bias -> gradient, scalar, update: " + gradient + ", " + scalar +", " + update);
	}
	
	
	public double getPartialPosteriors(AVPair[] x, int label) {
		
		double posterior = 0.0;
		
		for (int i = 0; i < x.length; i++) {
			
			int hi = fh.getIndex(label,  x[i].index); 
			int sign = fh.getSign(label, x[i].index);
			posterior += (x[i].value *sign) * (1/this.scalararray[label]) * this.w[hi];
		}
		
		posterior += (1/this.scalararray[label]) * this.bias[label]; 
		posterior = s.value(posterior);		
		
		return posterior;
	}
	
	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {

		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();
		
		int treeIndex = 0;
		
		while(!this.tree.isLeaf(treeIndex)) {

			AbstractLearner.numberOfInnerProducts++;
			double p = getPartialPosteriors(x, treeIndex);
			
			if(p < 0.5) {
				treeIndex = this.tree.getChildNodes(treeIndex).get(1);
			} else {
				treeIndex = this.tree.getChildNodes(treeIndex).get(0);
			}
		}

		positiveLabels.add(new EstimatePair(this.tree.getLabelIndex(treeIndex), 1.0));
		
		return positiveLabels;
	}

	
	public void save(String fname) {
		}

	public void load(String fname) {
	}
	
	
	
}