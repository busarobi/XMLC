package Learner;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Properties;
import java.util.Random;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.EstimatePair;
import Data.Instance;
import Data.NodeComparatorPLT;
import Data.NodePLT;
import IO.DataManager;
import preprocessing.FeatureHasherFactory;
import util.CompleteTree;
import util.HuffmanTree;
import util.PrecomputedTree;

public class DeepPLT extends PLT {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(DeepPLT.class);
	
	
	protected double[][] w = null;
	protected double[][] hiddenWeights = null;	
	protected int hiddendim = 100;
	
	transient protected int[] Tarrayhidden = null;
	protected double[] scalararrayhidden = null;
	protected double[] updatevec = null;
	
	protected String hiddenVectorsFile = null;
	
	public DeepPLT(Properties properties) {
		super(properties);
		
		System.out.println("#####################################################" );
		System.out.println("#### Learner: DeepPLT" );
		// learning rate

		this.hiddendim = Integer.parseInt(this.properties.getProperty("hiddendim", "100")); 
		logger.info("#### Number of hidden dimension: " + this.hiddendim );
		
		this.hiddenVectorsFile = this.properties.getProperty("hiddenvectorsFile", null);
		logger.info("#### hidden vectors file name " + this.hiddenVectorsFile );
		
		System.out.println("#####################################################" );

	}

	public void printParameters() {
		super.printParameters();
		logger.info("#### Number of hidden dimensions: " + this.hiddendim );
	}
	
	
	@Override
	public void allocateClassifiers(DataManager data) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();
		
		switch (this.treeType){
			case CompleteTree.name:
				this.tree = new CompleteTree(this.k, this.m);
				break;
			case PrecomputedTree.name:
				this.tree = new PrecomputedTree(this.treeFile);
				break;
			case HuffmanTree.name:
				this.tree = new HuffmanTree(data, this.treeFile);
				break;
			default:
				System.err.println("Unknown tree type!");
				System.exit(-1);
		}
		this.t = this.tree.getSize(); 

		logger.info( "#### Num. of labels: " + this.m + " Dim: " + this.d );
		logger.info( "#### Num. of inner node of the trees: " + this.t  );
		logger.info("#####################################################" );
			
		this.fh = FeatureHasherFactory.createFeatureHasher(this.hasher, fhseed, this.hd, 1);
		
		logger.info( "Allocate the learners..." );

		Random r = new Random();
		this.hiddenWeights = new double[this.hd][];
		for(int i = 0; i < this.hd; i++ ) {
			this.hiddenWeights[i] = new double[this.hiddendim];
			for(int j = 0; j < this.hiddendim; j++ ){
				this.hiddenWeights[i][j] = (2.0* r.nextDouble()-1); 
			}
		}
		
		this.w = new double[this.t][];
		for(int i = 0; i < this.t; i++ ) {
			this.w[i] = new double[this.hiddendim];
			for(int j = 0; j < this.hiddendim; j++ ){
				this.w[i][j] =  (2.0*r.nextDouble()-1);
			}
		}
		
		
		this.thresholds = new double[this.t];
		this.bias = new double[this.t];
		
		for (int i = 0; i < this.t; i++) {
			this.thresholds[i] = 0.5;
		}

		this.Tarray = new int[this.t];
		this.scalararray = new double[this.t];
		Arrays.fill(this.Tarray, 1);
		Arrays.fill(this.scalararray, 1.0);

		this.Tarrayhidden = new int[this.hd];
		this.scalararrayhidden = new double[this.hd];
		Arrays.fill(this.Tarrayhidden, 1);
		Arrays.fill(this.scalararrayhidden, 1.0);		

		this.updatevec = new double[this.hiddendim];
		
		logger.info( "Done." );
	}

	@Override
	public void train(DataManager data) {		
		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: {} ({})", (ep + 1), this.epochs );

			while( data.hasNext() == true ){
				
				Instance instance = data.getNextInstance();
				double[] hiddenRepresentation = this.getHiddenRepresentation(instance.x);
				
				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();

				for (int j = 0; j < instance.y.length; j++) {

					int treeIndex = this.tree.getTreeIndex(instance.y[j]); // + traindata.m - 1;
					positiveTreeIndices.add(treeIndex);

					while(treeIndex > 0) {

						treeIndex = this.tree.getParent(treeIndex); // Math.floor((treeIndex - 1)/2);
						positiveTreeIndices.add(treeIndex);

					}
				}

				if(positiveTreeIndices.size() == 0) {

					negativeTreeIndices.add(0);

				} else {

					
					for(int positiveNode : positiveTreeIndices) {
						
						if(!this.tree.isLeaf(positiveNode)) {
							
							for(int childNode: this.tree.getChildNodes(positiveNode)) {
								
								if(!positiveTreeIndices.contains(childNode)) {
									negativeTreeIndices.add(childNode);
								}
								
							}
							
						}
						
					}
				}
				
				Arrays.fill( this.updatevec, 0.0 );
				
				//logger.info("Negative tree indices: " + negativeTreeIndices.toString());
				for(int j:positiveTreeIndices) {

					double posterior = getPartialPosteriors(hiddenRepresentation,j);
					double inc = -(1.0 - posterior); 

					updatedTreePosteriors(hiddenRepresentation, j, inc);					
				}

				for(int j:negativeTreeIndices) {

					if(j >= this.t) logger.info("ALARM");

					double posterior = getPartialPosteriors(hiddenRepresentation,j);
					double inc = -(0.0 - posterior); 
					
					updatedTreePosteriors(hiddenRepresentation, j, inc);					
				}
				
				updateHiddenRepresentation( instance );
				
				this.T++;

				if ((this.T % 100000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ this.T );
				}
			}
			data.reset();
			
			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			if (this.hiddenVectorsFile != null )
				this.writeHiddenVectors(this.hiddenVectorsFile);
		}
		
	}

	protected void updateHiddenRepresentation( Instance instance ) {		

		double sum = 0.0;
		for(int j = 0; j < instance.x.length; j++ ){
			sum += instance.x[j].value;
		}
		if (sum > 0.000001) {
			sum = 1.0/sum;
		}
		
				
		for( int i = 0; i < instance.x.length; i++ ) {
			int hi = fh.getIndex(1,  instance.x[i].index); 
			//int sign = fh.getSign(1, instance.x[i].index);
		
			
			this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarrayhidden[hi]);
			this.Tarrayhidden[hi]++;
			this.scalararrayhidden[hi] *= (1 + this.learningRate * this.lambda);
			
			
			for(int j = 0; j < this.hiddendim; j++ ){
				//double gradient = this.scalararrayhidden[hi] * inc * this.w[ind][j] * sum * instance.x[i].value;
				double gradient = this.scalararrayhidden[hi] * this.updatevec[j] * sum * instance.x[i].value;
				double update = (this.learningRate * gradient);// / this.scalar;		
				this.hiddenWeights[hi][j] -= update; 				
			}
			
		}
	}

	
	protected double[] getHiddenRepresentation( AVPair[] x ) {
		double[] hiddenRepresentation = new double[this.hiddendim];
		// aggregate the the word2vec representation
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			int hi = fh.getIndex(1,  x[i].index); 
			//int sign = fh.getSign(1, x[i].index);
			
			for(int j = 0; j < this.hiddendim; j++ ){
				hiddenRepresentation[ j ] +=  x[i].value * (1.0 / this.scalararrayhidden[hi]) * this.hiddenWeights[hi][j];
				sum += x[i].value;
			}
			
		}
		
		if (sum > 0.0000001 ) {
			for(int j = 0; j < this.hiddendim; j++ ){
				hiddenRepresentation[ j ] /= sum;			
			}
		}
		
		return hiddenRepresentation;
	}
	
	
	protected void updatedTreePosteriors( double[] x, int label, double inc) {
			
		this.learningRate = this.gamma / (1 + this.gamma * this.lambda * this.Tarray[label]);
		this.Tarray[label]++;
		this.scalararray[label] *= (1 + this.learningRate * this.lambda);
		
		int n = x.length;
		
		for(int i = 0; i < this.hiddendim; i++) {
			double gradient = this.scalararray[label] * inc * x[i];
			double update = (this.learningRate * gradient);// / this.scalar;		
			this.w[label][i] -= update; 
			
			this.updatevec[i] += inc * this.w[label][i];
		}
		
		double gradient = this.scalararray[label] * inc;
		double update = (this.learningRate * gradient);//  / this.scalar;		
		this.bias[label] -= update;		
	}
	
	
	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = 1.0;

		double[] hiddenRepresentation = this.getHiddenRepresentation(x);
		
		int treeIndex = this.tree.getTreeIndex(label);

		posterior *= getPartialPosteriors(hiddenRepresentation, treeIndex);

		while(treeIndex > 0) {

			treeIndex = this.tree.getParent(treeIndex); //Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(hiddenRepresentation, treeIndex);

		}
		//if(posterior > 0.5) logger.info("Posterior: " + posterior + "Label: " + label);
		return posterior;
	}

	public double getPartialPosteriors(double[] x, int label) {
		double posterior = 0.0;
		
		
		for (int i = 0; i < this.hiddendim; i++) {			
			posterior += x[i] * (1.0/this.scalararray[label]) * this.w[label][i];
		}
		
		posterior += (1.0/this.scalararray[label]) * this.bias[label]; 
		posterior = s.value(posterior);		
		
		return posterior;

	}
	
	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		double[] hiddenRepresentation = this.getHiddenRepresentation(x);
		
		TreeSet<EstimatePair> positiveLabels = new TreeSet<EstimatePair>();

	    int foundTop = 0;
	    
	    NodeComparatorPLT nodeComparator = new NodeComparatorPLT();

		PriorityQueue<NodePLT> queue = new PriorityQueue<NodePLT>(11, nodeComparator);

		queue.add(new NodePLT(0,1.0));
		
		while(!queue.isEmpty() && (foundTop < k)) {

			NodePLT node = queue.poll();

			double currentP = node.p;
			
			if(!this.tree.isLeaf(node.treeIndex)) {
				
				for(int childNode: this.tree.getChildNodes(node.treeIndex)) {
					queue.add(new NodePLT(childNode, currentP * getPartialPosteriors(hiddenRepresentation, childNode)));
				}
				
			} else {
				
				positiveLabels.add(new EstimatePair(this.tree.getLabelIndex(node.treeIndex), currentP));
				foundTop++;
				
			}
		}

		return positiveLabels;
	}

	public void writeHiddenVectors( String outfname ) 
	{
		try{
			BufferedWriter bf = new BufferedWriter(new FileWriter(outfname) );
			
			for( int i = 0; i<this.d; i++)
			{
				int hi = fh.getIndex(1,  i);
				
				bf.write( i + "," +  hi );
				// labels
				for(int j=0; j< this.hiddendim; j++ ) {
					//logger.info(data.y[i][j]);
					bf.write(  "," + this.hiddenWeights[hi][j]  );
				}		
				
				bf.write( "\n" );
			}
			
			bf.close();
		} catch ( IOException e ) {
			System.out.println(e.getMessage());
		}
	}
	
	
}
	
	
