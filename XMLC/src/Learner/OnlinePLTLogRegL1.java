package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Random;

import Data.AVPair;
import Data.AVTable;
import IO.PerformanceMeasures;
import IO.Result;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.classifiers.linear.LinearSGD;
import jsat.io.LIBSVMLoader;
import jsat.lossfunctions.LogisticLoss;
import jsat.math.decayrates.PowerDecay;

public class OnlinePLTLogRegL1 {
	protected ClassificationDataSet traindata = null;
	protected ClassificationDataSet testdata = null;
	
	protected String trainfileName = null;
	protected String testfileName = null;
	
	public int[][] trainy;
	public int[][] testy;

	//public int n;           // number of samples
	public int trainm = -1;           // number of labels
	//public int d = -1;           // number of features
	public int t;
	protected int epochs = 100;
	protected double threshold = 0.01;
	
	
	protected BaseUpdateableClassifier[] learners = null;
	
	public void createClassifier()
	{		
		
		// create classifiers
		this.t = 2 * trainm - 1;

		this.learners = new BaseUpdateableClassifier[this.t];
		
		double lambda0 = 0.000001;
		//double lambda1 = 0.0001;
		double lambda1 = 0.0; // no L1Q vector
		double eta = 0.5;
		
		System.out.println( "    --> eta: " + eta + " Lambda0 " + lambda0  + " Lambda1: " + lambda1 );		
		
		for (int i = 0; i < this.t; i++) {
			LogisticLoss loss = new LogisticLoss();
			PowerDecay pd = new PowerDecay(1, 0.1);
			
			learners[i] = new LinearSGD( loss, eta, pd, lambda0, lambda1);
			learners[i].setUp(null, traindata.getNumFeatures(), traindata.getPredicting());
		}

		// create classifiers
				
	}
	
	
	public void train()
	{
		for (int ep = 0; ep < this.epochs; ep++) {

			System.out.println("--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			
			// random permutation
			ArrayList<Integer> indiriectIdx = new ArrayList<Integer>();
			for (int i = 0; i < this.traindata.getSampleSize(); i++) {
				indiriectIdx.add(new Integer(i));
			}

			Collections.shuffle(indiriectIdx);

			for (int i = 0; i < traindata.getSampleSize(); i++) {

				if ((i % 100000) == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + traindata.getSampleSize() + ")" );
					
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
				}
		
				int currIdx = indiriectIdx.get(i);

				HashSet<Integer> positiveTreeIndices = new HashSet<Integer>();
				HashSet<Integer> negativeTreeIndices = new HashSet<Integer>();
					
				//System.out.print("Positive Labels: ");
				int tl = 0;
				if ( trainy[currIdx] != null ) tl = trainy[currIdx].length;
				
				for (int j = 0; j < tl; j++) {
						
					//if(j == traindata.y[currIdx].length - 1) 
					//	System.out.println(traindata.y[currIdx][j]);
					//else
					//	System.out.print(traindata.y[currIdx][j] + ", ");
					
					int treeIndex = trainy[currIdx][j] + trainm - 1;
					positiveTreeIndices.add(treeIndex);
						
					while(treeIndex > 0) {
					
						treeIndex = (int) Math.floor((treeIndex - 1)/2);
						positiveTreeIndices.add(treeIndex);
					
					}	
				}
					
				if(positiveTreeIndices.size() == 0) {
					
					negativeTreeIndices.add(0);
					
				} else {
						
					PriorityQueue<Integer> queue = new PriorityQueue<Integer>();
					queue.add(0);
					
					//System.out.print("Positive tree indices: ");
					
					while(!queue.isEmpty()) {
							
						int node = queue.poll();
						int leftchild = 2 * node + 1;
						int rightchild = 2 * node + 2;
						
						Boolean left = false, right = false;
					
						if(positiveTreeIndices.contains(leftchild)) {
							queue.add(leftchild);
							left = true;
						}

						if(positiveTreeIndices.contains(rightchild)) {
							queue.add(rightchild);
							right = true;
						}

						if(left == true && right == false) {
							negativeTreeIndices.add(rightchild);
						}
						
						if(left == false && right == true) {
							negativeTreeIndices.add(leftchild);
						}
						
						if(queue.isEmpty()) {
						//	System.out.println(node);
						} else {
						//	System.out.print(node + ", ");
						}
						
					}
				}
					
				//System.out.println("Negative tree indices: " + negativeTreeIndices.toString());
				
				DataPoint p = this.traindata.getDataPoint(currIdx);
				
				for(int j:positiveTreeIndices) {					
					//double posterior = getPartialPosteriors(p,j);
					learners[j].update(p, 1 );									
				}
					
				for(int j:negativeTreeIndices) {
					
					if(j >= this.t) System.out.println("ALARM");					
					learners[j].update(p, 0 );					
				}
				
			}
			
			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );			
		}		
	}

	public double getPartialPosteriors(DataPoint p, int label) 
	{	
		double posterior = 0.0;
		CategoricalResults result = learners[label].classify(p);
		posterior = result.getProb(1); 		
		return posterior;
	}


	public double getPosteriors(DataPoint p, int label) {
		double posterior = 1.0;
		
		
		int treeIndex = label + trainm - 1;
		
		posterior *= getPartialPosteriors(p, treeIndex);
		
		while(treeIndex > 0) {			
			treeIndex = (int) Math.floor((treeIndex - 1)/2);
			posterior *= getPartialPosteriors(p, treeIndex);
		
		}	
				
		return posterior;
	}
	
	
	
    public ClassificationDataSet loadData( String fileName, int vecLength )
    {
    	ClassificationDataSet ds = null;
		//System.out.print( "Loading file...(" + fileName + ".data)" );
		File f = new File(fileName + ".data");		
		try {
			ds = LIBSVMLoader.loadC(f,0.5, vecLength);
		} catch (FileNotFoundException e) {
			System.out.println( "File not found!!");
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println( "File format problem!!");
			e.printStackTrace();
		}
		return ds;		
    }
	
	
	public void loadTrainFile()
	{
		traindata = loadData( trainfileName, -1 );
		System.out.println( "Training data -->" + trainfileName + ".data\n  num x dim: " 
		           + traindata.getSampleSize() + " x "+ traindata.getNumFeatures() );
		
		trainy = loadLabels( trainfileName, traindata.getSampleSize() );
		System.out.println( "Training label data -->  num:" + 
		          " " + trainy.length  );
		
		for( int i = 0; i< trainy.length; i++ )
		{
			if (trainy[i] != null ){
				for( int j = 0; j < trainy[i].length; j++ )
					trainm = Math.max( trainm, trainy[i][j] );
			}
		}
		trainm = trainm + 1;
	}

	public void loadTestFile()
	{
		testdata = loadData( testfileName, this.traindata.getNumFeatures() );
		System.out.println( "Test data -->" + testfileName + ".data\n  num x dim: " 
		           + testdata.getSampleSize() + " x "+ testdata.getNumFeatures() );
		
		testy = loadLabels( testfileName, testdata.getSampleSize() );
		System.out.println( "Test label data -->  num:" + 
		          " " + testy.length  );		
	}
	
	
	
	public int[][] loadLabels( String fname, int numOfInstance )
	{
		int[][] labels = null;
		try {
			String completefname = fname + ".labs";
			System.out.println( "--> Label file: " +  completefname );
			FileInputStream fis = new FileInputStream(completefname);
		 
			BufferedReader br = new BufferedReader(new InputStreamReader(fis));
	 
			String line = null;
			labels = new int[numOfInstance][];
			
			int i = 0;
			while ((line = br.readLine()) != null)	{
				line = line.trim();
				if (!line.isEmpty())
				{
					String[] ss =  line.split(",");
					labels[i] = new int[ss.length];
	
					//System.out.println(line + " length: " + ss.length);				
					
					for(int j = 0; j< ss.length; j++ ){
						labels[i][j] = Integer.parseInt(ss[j].trim());						
					}
				}				
				i++;
			}
			
			br.close();		
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}
		
		return labels;
	}
	
	
	public void outputPosteriors( ClassificationDataSet data, String fname )
	{
		System.out.print( "--> Output posteriors: " + fname +"...");
		
		this.threshold = 0.0001;
		
	    BufferedWriter output = null;

	    try {
            File file = new File(fname);
            output = new BufferedWriter(new FileWriter(file));

    		for (int i = 0; i < data.getSampleSize(); i++) {
    			
    			HashSet<Integer> predictedLabels = this.getPositiveLabels(data.getDataPoint(i));
    			
    			for (Integer pl : predictedLabels ){
        			double posterior = this.getPosteriors(data.getDataPoint(i), pl);
        			output.write( pl + ":" + posterior + " ");    				
    			}
    			
    			output.write("\n");
    		}
            
            output.close();
        } catch ( IOException e ) {
            e.printStackTrace();
        } 
	    
	    System.out.println( " Done." );
	}
	
	public Result test() {
		double[][] posteriors = new double[this.testdata.getSampleSize()][];
		
		for (int i = 0; i < this.testdata.getSampleSize(); i++) {
			posteriors[i] = new double[this.trainm];
			for (int j = 0; j < this.trainm; j++) {
				posteriors[i][j] = this.getPosteriors(this.testdata.getDataPoint(i), j);
			}
		}

		Result res = new Result(posteriors, testy, this.testdata.getSampleSize(), trainm);
		return res;
	}
	
	public HashSet<Integer>[] getPredictedLabels(ClassificationDataSet testdata) {
		HashSet<Integer>[] predictedLabels = new HashSet[testdata.getSampleSize()];
				
		for (int i = 0; i < testdata.getSampleSize(); i++) {
			
			predictedLabels[i] = this.getPositiveLabels(testdata.getDataPoint(i));
	
		}
		return predictedLabels;
	}
	
	
	public HashSet<Integer> getPositiveLabels(DataPoint x) {
		HashSet<Integer> positiveLabels = new HashSet<Integer>();

		class Node {

			int treeIndex;
			double p;
			
			Node(int treeIndex, double p) {
				this.treeIndex = treeIndex;
				this.p = p;
			}
			
			public String toString() {
				return new String("(" + this.treeIndex + ", " + this.p + ")");
			}
		};
		
		class NodeComparator implements Comparator<Node> {
	        public int compare(Node n1, Node n2) {
	        	return (n1.p > n2.p) ? 1 : -1;
	        }
	    } ;
		
	    NodeComparator nodeComparator = new NodeComparator();
		
		PriorityQueue<Node> queue = new PriorityQueue<Node>(this.t,nodeComparator);
		
		
		queue.add(new Node(0,1.0));
		
		
		while(!queue.isEmpty()) {
				
			Node node = queue.poll();
			
			double currentP = node.p * getPartialPosteriors(x, node.treeIndex);
			
			if(currentP > this.threshold) {
			
				if(node.treeIndex < this.trainm - 1) {
					int leftchild = 2 * node.treeIndex + 1;
					int rightchild = 2 * node.treeIndex + 2;
					
					queue.add(new Node(leftchild, currentP));
					queue.add(new Node(rightchild, currentP));
					
				} else {

					positiveLabels.add(node.treeIndex - this.trainm + 1);
				
				}
			}
		}
		
		//System.out.println("Predicted labels: " + positiveLabels.toString());
		
		return positiveLabels;
	}
	
	
	
	////////////////////////////////////////////////////////////////////////////////////////////
	// START: Performance metrics
	////////////////////////////////////////////////////////////////////////////////////////////
	
	public double[] computePerf(ClassificationDataSet data, int[][] y, int m ) {
		System.out.println("--> Computing Hamming loss...");

		double macroF = 0.0;
		
		int[] tp = new int[m];
		int[] yloc = new int[m];
		int[] haty = new int[m];
		
		double HL = 0.0;
		int n = data.getSampleSize();
		
		int numOfPositives = 0;
		
		for(int i = 0; i < n; i++ ) {
			
			
			
			if (y[i] == null ) continue;
			
			HashSet<Integer> predictedLabels = this.getPositiveLabels(data.getDataPoint(i));
			numOfPositives += predictedLabels.size();
			
			
			// Hamming
			int tploc = 0, fn = 0;
			int tmpsize = 0;
			for(int trueLabel: y[i]) {
				if (trueLabel< m) { // this label was seen in the training
					tmpsize++;
					if(predictedLabels.contains(trueLabel)) {
						tploc++;
					} else {
						fn++;
					}
				}
			}
			
			HL += (fn + (tmpsize - tploc));

			
			
			// F-score
			for(int trueLabel: y[i]) {
				if (trueLabel>= m) continue; // this label is not seen in the training
				
				if(predictedLabels.contains(trueLabel)) {
					tp[trueLabel]++;
				}
				yloc[trueLabel]++;
			}

			for(int predictedLabel: predictedLabels) {
				haty[predictedLabel]++;
			}
			
			
			if ((i % 10000) == 0) {
				System.out.println( "----->\tSample: "+ i +" (" + data.getSampleSize() + ")" );
				
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				System.out.println("\t\t" + dateFormat.format(date));
				System.out.println( "\t\t Avg. num. of predicted positives: " + (numOfPositives / (double) (i+1)) );				
			}
			
		}
		
		HL = HL / ((double)(n * m));
		
		int presentedfeatures = 0;
		for(int i = 0; i < m; i++) {
			double denum = (double) (yloc[i] + haty[i]);
			if (yloc[i]>0.0)
			{
				macroF += (2.0 * tp[i])/denum;
				presentedfeatures++;
			}
		}
		
		macroF = macroF/((double) presentedfeatures);
		
		double[] arr = new double[2];
		arr[0] = HL;
		arr[1] = macroF;
		
		return arr;
	}
	

	////////////////////////////////////////////////////////////////////////////////////////////
	// END: Performance metrics
	////////////////////////////////////////////////////////////////////////////////////////////

	
	public static void main(String[] args) {
//		String trainfileName = "../data/mediamill/train-exp1.svm";
//		String testfileName = "../data/mediamill/test-exp1.svm";


		//String trainfileName = "../data/test/test.svm";
		//String testfileName = "../data/test/test.svm";

		
		OnlinePLTLogRegL1 learner = new OnlinePLTLogRegL1();
		//learner.trainfileName = "/Users/busarobi/work/XMLC/data/mediamill/train-exp1";
		//learner.testfileName = "/Users/busarobi/work/XMLC/data/mediamill/test-exp1";

		learner.trainfileName = args[0];
		learner.testfileName = args[1];		

		
		learner.loadTrainFile();
		learner.loadTestFile();
	
		learner.createClassifier();
		learner.train();
		
		//Result result = learner.test();
		//System.out.println("Hamming loss: " + result.getHL());

		//PerformanceMeasures pm = new PerformanceMeasures();
		//HashSet<Integer>[] predictedlabels = learner.getPredictedLabels(learner.testdata);
		//System.out.println("##### Hamming loss: " + pm.computeHammingLoss(predictedlabels, learner.testy, learner.testdata.getSampleSize(), learner.trainm));
		//System.out.println("##### Macro-F: " + pm.computeMacroF(predictedlabels, learner.testy, learner.testdata.getSampleSize(), learner.trainm));		
		
//		 double[] perf = learner.computePerf(learner.testdata, learner.testy, learner.trainm);
//		 System.out.println("##### Hamming loss: " + perf[0]);
//		 System.out.println("##### Macro-F: " + perf[1] );
		
		learner.outputPosteriors(learner.traindata, learner.trainfileName+".post");
		learner.outputPosteriors(learner.testdata, learner.testfileName+".post");
		
		
		 ArrayList<double[]> resthres = new ArrayList<double[] >(); 
		
		 learner.threshold = 0.05;
		 for (int i=0; i < 10; i++ ) { 
			 
			 System.out.println( "##############################################################");
			 System.out.println( "##### Threshold: " + learner.threshold );
			 
			 double[] perf = learner.computePerf(learner.testdata, learner.testy, learner.trainm);
			 System.out.println("##### Hamming loss: " + perf[0]);
			 System.out.println("##### Macro-F: " + perf[1] );
			 
			 resthres.add(perf);
			 
			 learner.threshold += 0.05;
		 }
		 
		double th = 0.05;
		 for (int i=0; i < resthres.size(); i++ ) {
			 System.out.println( th + "," + resthres.get(i)[0] + "," +resthres.get(i)[1] );
			 th += 0.05;
		 }
		 
	}

}
