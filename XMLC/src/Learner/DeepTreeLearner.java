package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer.EmptyClusterStrategy;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;

import Data.AVPair;
import Data.EstimatePair;
import Data.Instance;
import IO.BatchDataManager;
import IO.DataManager;
import IO.Evaluator;
import IO.ReadProperty;
import util.PrecomputedTree;

public class DeepTreeLearner extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(DeepTreeLearner.class);
	
	protected int hiddendim = 0;
	protected String hiddenVectorsFile;
	protected String hiddenLabelVectorsFile;
	protected int numOfThreads = 1;
	
	protected int k = 2;
	protected String treeFile = null;
	protected PrecomputedTree tree = null;
	protected int treebuildingepochs = 1;
	
	protected DataManager traindata = null;
	protected double[][] hiddenWeights = null;		
	protected double[][] hiddenLabelRep = null;
	protected ParallelDeepPLT learner = null;

	protected int kForRFdistance = -1;

	protected ArrayList<PrecomputedTree> treeArray  = new ArrayList<PrecomputedTree>(); 
	protected ArrayList<ArrayList<Double>> treeDistances = new ArrayList<ArrayList<Double>>();
	
	
	
	public DeepTreeLearner(Properties properties) {
		super(properties);
		System.out.println("#####################################################");
		System.out.println("#### Learner: DeepTreeLearner");

		// k-ary tree
		this.k = Integer.parseInt(this.properties.getProperty("k", "2"));
		logger.info("#### k (order of the tree): " + this.k );

		// tree file name
		this.treeFile = this.properties.getProperty("treeFile", null);
		logger.info("#### tree file name " + this.treeFile );
		
		
		this.hiddendim = Integer.parseInt(this.properties.getProperty("hiddendim", "100"));
		logger.info("#### Number of hidden dimension: " + this.hiddendim);

		this.hiddenVectorsFile = this.properties.getProperty("hiddenvectorsFile", null);
		logger.info("#### hidden vectors file name " + this.hiddenVectorsFile);

		this.hiddenLabelVectorsFile = this.properties.getProperty("hiddenlabelvectorsFile", null);
		logger.info("#### hidden label vectors file name " + this.hiddenLabelVectorsFile);

		// epochs
		this.treebuildingepochs = Integer.parseInt(this.properties.getProperty("treeepochs", "1"));
		logger.info("#### epochs: " + this.treebuildingepochs );
		
		//
		this.numOfThreads = Integer.parseInt(this.properties.getProperty("numThreads", "4"));
		logger.info("#### num of threads: " + this.numOfThreads);

		// depth for the RF distance up to a depth
		this.kForRFdistance = Integer.parseInt(this.properties.getProperty("kForRFdistance", "-1"));
		logger.info("#### k for RF distance: " + this.kForRFdistance);

		System.out.println("#####################################################");

	}

	@Override
	public void allocateClassifiers(DataManager data) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();
	}

	protected void treeBuilding() {
		ArrayList<Integer> indices = new ArrayList<Integer>();
		for( int i = 0; i < this.m; i++) indices.add(i);
		this.treeIndices = new ArrayList<Integer>();
		
		this.treeIdx = 0;
		treeIndices.add(0); // root
		treeIndices.add(0); // root
		treeIndices.add(0); // root
		
		hierarchicalClustering(0, indices);
		this.tree = new PrecomputedTree( treeIndices );
	}
	
	private int treeIdx = -1;
	private ArrayList<Integer> treeIndices = null;
	
	protected void hierarchicalClustering(int parent, ArrayList<Integer> indices ){
		if (indices.size() >= this.k){
			
						
			//logger.info("Clustering... (size: " + indices.size() + ")" + indices.toString() );
			//logger.info("Clustering... (size: " + indices.size() + ")");
			List<ClusteringWrapper> clusterInput = new ArrayList<ClusteringWrapper>();
			for (int i = 0; i < indices.size(); i++ )
			    clusterInput.add(new ClusteringWrapper(this.hiddenLabelRep[indices.get(i)], indices.get(i)));
			
			// initialize a new clustering algorithm. 
			// we use KMeans++ with 10 clusters and 10000 iterations maximum.
			// we did not specify a distance measure; the default (euclidean distance) is used.
			KMeansPlusPlusClusterer<ClusteringWrapper> clusterer 
					= new KMeansPlusPlusClusterer<ClusteringWrapper>(this.k, 1000,
																	 new EuclideanDistance(), new JDKRandomGenerator(), EmptyClusterStrategy.LARGEST_VARIANCE );
																	 //new Random(), 
																	 //EmptyClusterStrategy.LARGEST_VARIANCE  );
			//System.out.println(clusterer.getEmptyClusterStrategy());
			List<CentroidCluster<ClusteringWrapper>> clusterResults = null;
			try {
				clusterResults = clusterer.cluster(clusterInput);
			} catch (ConvergenceException e) {
				logger.info(e.getMessage());
				addIndicesToTree(parent, indices );
				return;
			}
	
			
			// remove empty clusters
			ArrayList<CentroidCluster<ClusteringWrapper> >  filteredClusters = new ArrayList<CentroidCluster<ClusteringWrapper> >();
			for (int i=0; i<clusterResults.size(); i++) {
				CentroidCluster<ClusteringWrapper> currentCluster = clusterResults.get(i);
				int size = currentCluster.getPoints().size();
				if (size>0){
					filteredClusters.add(currentCluster);
				}
			}			
			
			// output cluster size
			ArrayList<Integer> sizeOfClusters = new ArrayList<Integer>();
			//logger.info( "Number of clusters: {}", filteredClusters.size() );
			for (int i=0; i<filteredClusters.size(); i++) {
				CentroidCluster<ClusteringWrapper> currentCluster = filteredClusters.get(i);
				int size = currentCluster.getPoints().size();
				sizeOfClusters.add(size);							
			}			
			//logger.info( "-------> Size of cluster: " + sizeOfClusters.toString() );
			
			
			if (filteredClusters.size()<=1) {
				addIndicesToTree(parent, indices );
			} else { // recursion
				int currentIdx = ++treeIdx;
				this.treeIndices.add(parent);
				this.treeIndices.add(currentIdx);
				this.treeIndices.add(0);

				for (int i=0; i<filteredClusters.size(); i++) {		    			
					ArrayList<Integer> childIndices = new ArrayList<Integer>();
					CentroidCluster<ClusteringWrapper> currentCluster = filteredClusters.get(i);
					
					if( currentCluster.getPoints().size()>0){
						for (ClusteringWrapper locationWrapper : currentCluster.getPoints())
							childIndices.add(locationWrapper.getInd());
						
						this.hierarchicalClustering(currentIdx, childIndices);
					}				
				}
			}
		} else {			
			//logger.info("Cluster ... (" + indices.size() + ")" );
			addIndicesToTree(parent, indices );
		}
	}
	
	private void addIndicesToTree(int parent, ArrayList<Integer> indices ){
		for( int i = 0; i < indices.size(); i++ ){
			int currentIdx = ++treeIdx;
			this.treeIndices.add(parent);
			this.treeIndices.add(currentIdx);
			this.treeIndices.add(0);				

			this.treeIndices.add(currentIdx);
			this.treeIndices.add(indices.get(i));
			this.treeIndices.add(1);				
		}

	}

	protected void buildLabelHiddenPresenation( DataManager data) {
		logger.info("Computing hidden label representations...");
		this.hiddenLabelRep = new double[this.m][];
		for (int i = 0; i < this.m; i++) {
			this.hiddenLabelRep[i] = new double[this.hiddendim];
		}

				
		int[] numelOfSums = new int[this.m];
		int ni = 0;
		while( data.hasNext() == true ){			
			Instance instance = data.getNextInstance();
			ni++;
			if (ni % 100000 == 0)
				logger.info("Processed instances " + ni );
			
			for(int labi = 0; labi < instance.y.length; labi++ ){				
				for( int xi = 0; xi < instance.x.length; xi++ ){
					numelOfSums[instance.y[labi]]++;
					for( int hdi = 0; hdi < this.hiddendim; hdi++ ){
						this.hiddenLabelRep[instance.y[labi]][hdi] += this.hiddenWeights[instance.x[xi].index][hdi];
					}
				}
			}
		}
		
		for (int i = 0; i < this.m; i++) {
			if (numelOfSums[i]>0){
				for( int hdi = 0; hdi < this.hiddendim; hdi++ ){
					this.hiddenLabelRep[i][hdi] /= numelOfSums[i];
				}
			}
		}	
		
		data.reset();
		logger.info("Hidden label representations are computed...");
	}
	
	
	public void writeHiddenLabelVectors(String outfname) {
		try {
			BufferedWriter bf = new BufferedWriter(new FileWriter(outfname));

			for (int i = 0; i < this.m; i++) {
				

				bf.write(i + "" );
				// labels
				for (int j = 0; j < this.hiddendim; j++) {
					// logger.info(data.y[i][j]);
					bf.write("," + this.hiddenLabelRep[i][j]);
				}

				bf.write("\n");
			}

			bf.close();
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}	
	
	protected void readDeepRepresentationOfFeatures() {
		this.hiddenWeights = new double[this.d][];
		for (int i = 0; i < this.d; i++) {
			this.hiddenWeights[i] = new double[this.hiddendim];
		}
		
		try{
			//BufferedReader fp = new BufferedReader(new FileReader(this.hiddenVectorsFile));
			BufferedReader fp = new BufferedReader(new FileReader("/Users/busarobi/work/XMLC/Clustering/data/hiddenvectors_parallel_local.txt"));
			
			for(int i = 0; i < this.d; i++)
			{
				String line = fp.readLine();
				if(line == null) break;
								
				StringTokenizer st = new StringTokenizer(line,",");			
				int featIdx = Integer.parseInt(st.nextToken());
				
				st.nextToken();
				for(int j=0; j<this.hiddendim; j++ ){
					double val = Double.parseDouble(st.nextToken());
					this.hiddenWeights[featIdx][j] = val; 
				}

			}
			
			fp.close();
		} catch (IOException e ){
			System.out.println(e.getMessage());
		}
		logger.info("Hidden representation is loaded" );
	}
	
	@Override
	public void train(DataManager data) {

		this.learner = new ParallelDeepPLT(this.properties);
		this.learner.allocateClassifiers(data);
		this.learner.train(data);
		
		
		for (int ep = 0; ep < this.treebuildingepochs; ep++) {

			logger.info("#############################################################################");
			logger.info("##########################--> BEGIN of Tree learning Epoch: {} ({})", (ep + 1), this.treebuildingepochs);
			
			this.hiddenWeights = this.learner.getDeepRepresentation();
			//this.readDeepRepresentationOfFeatures();
			this.buildLabelHiddenPresenation( data);
			//this.writeHiddenLabelVectors(this.hiddenLabelVectorsFile);		
			this.treeBuilding();
			
			if (this.treeFile!=null) {
				String tmpTreeFile = this.treeFile + "_" + (ep+1) + ".txt";
				logger.info("Write tree to " + this.treeFile );
				this.tree.writeTree(tmpTreeFile);
			}
//			this.writeTreeIndices();
			
			
			this.learner = new ParallelDeepPLT(this.properties);
			this.learner.allocateClassifiers(data, this.tree);
						
			this.learner.train(data);
			logger.info("--> END of tree laerning epoch: " + (ep + 1) + " (" + this.treebuildingepochs + ")");
		}
		
	}

	
	protected void initepoch(DataManager data ) {
		this.learner = new ParallelDeepPLT(this.properties);
		this.learner.allocateClassifiers(data);
		this.learner.train(data);		
	}
	
	protected int currentEpoch = 0;
	protected void epochtrain(DataManager data) {

		currentEpoch++;

		logger.info("#############################################################################");
		logger.info("##########################--> BEGIN of Tree learning Epoch: {} ", currentEpoch);

		this.hiddenWeights = this.learner.getDeepRepresentation();
		
		this.buildLabelHiddenPresenation( data);		
		this.treeBuilding();
		
		this.computeTreeDistanceAndOutput();
		
		if (this.treeFile!=null) {
			String tmpTreeFile = this.treeFile + "_" + currentEpoch + ".txt";
			logger.info("Write tree to " + this.treeFile );
			this.tree.writeTree(tmpTreeFile);
		}
			
		this.learner = new ParallelDeepPLT(this.properties);
		this.learner.allocateClassifiers(data, this.tree);
			
		this.learner.train(data);
		logger.info("--> END of tree laerning epoch: {} ", currentEpoch );
		
	}
	
	protected void computeTreeDistanceAndOutput() {
		this.treeArray.add(this.tree);
		this.treeDistances.add(new ArrayList<Double>());
		
		for(int i = 0; i < this.treeArray.size()-1; i++) {
			double dist = this.tree.robinsonFouldsDistanceUpToDepthK(this.treeArray.get(i), this.kForRFdistance);
			this.treeDistances.get(i).add(dist);
		}
		
		for(int i = 0; i < this.treeDistances.size()-1; i++) {
			for( int j = 0; j < this.treeDistances.get(i).size(); j++ ){
				logger.info("Tree " + i + " Tree " + (j+1) + " " + this.treeDistances.get(i).get(j));
			}
		}
		
	}
	
	protected void writeTreeIndices(){
		File file;
		Writer wr;
		try {
			file = new File(this.treeFile.replace(".txt", "_raw.txt"));
			wr = new FileWriter(file);
			for(int i = 0; i < this.treeIndices.size(); i++ ){
				wr.write(this.treeIndices.get(i).toString());
				if ((i+1)%3==0)
					wr.write("\n");
				else
					wr.write(" ");
			}
			wr.close();
		} catch (java.io.IOException e) {
			e.printStackTrace();
		}
		
	}
	
	@Override
	public double getPosteriors(AVPair[] x, int label) {		
		return this.learner.getPosteriors(x, label);
	}

	public TreeSet<EstimatePair> getTopKEstimates(AVPair[] x, int k) {
		return this.learner.getTopKEstimates(x, k);
	}
	
	
	public static void main(String[] args) {
		Properties properties = ReadProperty.readProperty(args[0]);
		
		int numOfThreads = Integer.parseInt(properties.getProperty("numThreads", "4"));
		
		DeepTreeLearner learner = new DeepTreeLearner(properties);
		
		DataManager traindata = new BatchDataManager(properties.getProperty("TrainFile")); 
//		DataManager traindata = DataManager.managerFactory(properties.getProperty("TrainFile"), "Online" );
		learner.allocateClassifiers(traindata);
		

//		DataManager testdata = DataManager.managerFactory(properties.getProperty("TestFile"), "Online" );
		DataManager testdata = new BatchDataManager(properties.getProperty("TestFile") );

		// no tree building
		learner.initepoch(traindata);
		logger.info("#################### Evaluating the model (no tree learning) #############################");
		
		
		
		Map<String, Double> perftestpreck = Evaluator.computePrecisionAtk(learner, testdata, 5, numOfThreads );
		
		for (String perfName : perftestpreck.keySet()) {
			logger.info("##### Test " + perfName + ": " + perftestpreck.get(perfName));
		}	
		
		
		for (int ep = 0; ep < learner.treebuildingepochs; ep ++ ) {
			learner.epochtrain(traindata);
			
			logger.info("#################### Evaluating the model #############################");
			testdata.reset();
			perftestpreck = Evaluator.computePrecisionAtk(learner, testdata, 5, numOfThreads );
			
			for (String perfName : perftestpreck.keySet()) {
				logger.info("##### Test " + perfName + ": " + perftestpreck.get(perfName));
			}	
			
		}		
		
	}

	// wrapper class
	public static class ClusteringWrapper implements Clusterable {
	    private double[] points;
	    private int ind = -1;

	    public ClusteringWrapper(double[] data, int i) {
	        this.points = data;
	        this.ind = i;
	    }

	    public int getInd() {
	    	return this.ind;
	    }
	    
	    public double[] getPoint() {
	        return points;
	    }
	}
	
	
}
