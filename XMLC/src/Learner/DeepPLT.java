package Learner;

import java.util.Properties;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import IO.DataManager;
import preprocessing.FeatureHasher;
import util.Tree;

public class DeepPLT extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(DeepPLT.class);
	transient protected int t = 0;	
	protected Tree tree = null;
	
	protected int k = 2;
	protected String treeType = "Complete";
	protected String treeFile = null;

	transient protected int T = 1;
	//transient protected AVTable traindata = null;
	transient protected DataManager traindata = null;
	
	
	transient protected FeatureHasher fh = null;
	protected String hasher = "Universal";
	protected int fhseed = 1;
	protected int hd;
	
	protected double[] bias;
	protected double[] w = null;
	
	transient protected int[] Tarray = null;	
	protected double[] scalararray = null;

	protected double gamma = 0; // learning rate
	transient protected int step = 0;
	static Sigmoid s = new Sigmoid();
	transient protected double learningRate = 1.0;
	protected double scalar = 1.0;
	protected double lambda = 0.00001;
	protected int epochs = 1;	

	public DeepPLT(Properties properties) {
		super(properties);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void allocateClassifiers(DataManager data) {
		// TODO Auto-generated method stub

	}

	@Override
	public void train(DataManager data) {
	

	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		// TODO Auto-generated method stub
		return 0;
	}

}
