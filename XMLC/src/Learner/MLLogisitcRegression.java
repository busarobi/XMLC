package Learner;

import java.util.Random;

import Data.AVTable;
import IO.DataReader;
import IO.Evaluator;
import IO.Result;

public class MLLogisitcRegression extends AbstractLearner {
	protected int m = 0; // num of labels
	protected int d = 0;
	protected int epochs = 10;

	protected double[][] w = null;
	protected double[][] grad = null;
	protected double[] bias = null;
	protected double[] gradbias = null;

	protected double gamma = 0.33; // learning rate
	protected int step = 20;
	
	// uniform sampling of negatives, the number of negatives is r times more as many as positives
	protected int r = 1;            
	
	protected int T = 1;
	protected double delta = 0.01;
	protected AVTable traindata = null;
	
	// 0 = "vanila"
	protected int updateMode = 0;

	protected Random rand = new Random();

	@Override
	public void allocateClassifiers(AVTable data, String propertyFile) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;
		

		this.w = new double[this.m][];
		this.bias = new double[this.m];

		this.grad = new double[this.m][];
		this.gradbias = new double[this.m];

		for (int i = 0; i < this.m; i++) {
			this.w[i] = new double[d];
			this.grad[i] = new double[d];

			for (int j = 0; j < d; j++)
				this.w[i][j] = 2.0 * rand.nextDouble() - 1.0;

			this.bias[i] = 2.0 * rand.nextDouble() - 1.0;

		}

		// learning rate
		this.gamma = 0.1;
		// step size for learning rate
		this.step = 2000;
		this.T = 1;
		// decay of gradient
		this.delta = 0.01;


		
	}

	@Override
	public void train(AVTable data) {
		// TODO Auto-generated method stub
		
	}

	
	@Override
	public Evaluator test(AVTable data) {
		// TODO Auto-generated method stub
		return null;
	}
	

	public static void main(String[] args) throws Exception {
		// String fileName = "/Users/busarobi/work/XMLC/data/scene/scene_train";

		//String trainfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_train.svm";
		//String testfileName = "/Users/busarobi/work/XMLC/data/yeast/yeast_test.svm";

		String trainfileName = "/Users/busarobi/work/XMLC/data/mediamill/train-exp1.svm";
		String testfileName = "/Users/busarobi/work/XMLC/data/mediamill/test-exp1.svm";

		DataReader datareader = new DataReader(trainfileName);
		AVTable data = datareader.read();

		AbstractLearner learner = new MLLogisitcRegression();
		
		// train
		learner.allocateClassifiers(data,"MLLogReg.config");		
		learner.train( data );

		// test
		DataReader testdatareader = new DataReader(testfileName);
		AVTable testdata = testdatareader.read();

		//Result result = learner.test(testdata);
		//System.out.println("Hamming loss: " + result.getHL());


	}

	
	
}
