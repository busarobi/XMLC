package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;
import Data.DenseVectorExt;
import Data.SparseVectorExt;
import Learner.step.StepFunction;
import util.MasterSeed;

public class MLLogisticRegression extends AbstractLearner {
	private static final long serialVersionUID = 6048253331791813751L;

	private static Logger logger = LoggerFactory.getLogger(MLLogisticRegression.class);

	protected int epochs = 20;

	protected DenseVectorExt[] w = null;
	protected StepFunction[] stepfunctions;

	protected double gamma = 0; // learning rate
	protected int step = 0;
	
    //protected int OFOepochs = 1;

	protected boolean geomWeighting = true;

	protected int T = 1;
	protected AVTable traindata = null;

	// 0 = "vanila"
	protected int updateMode = 0;

	Random shuffleRand;
	
	transient Sigmoid s = new Sigmoid();


	public MLLogisticRegression(Properties properties, StepFunction stepfunction) {
		super(properties, stepfunction);
		shuffleRand = MasterSeed.nextRandom();

		logger.info("#####################################################" );
		logger.info("#### Leraner: LogReg" );

		// learning rate
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "10.0"));
		logger.info("#### gamma: " + this.gamma );

		// step size for learning rate
		this.step = Integer.parseInt(this.properties.getProperty("step", "2000") );
		logger.info("#### step: " + this.step );


		this.epochs = Integer.parseInt(this.properties.getProperty("epochs", "30"));
		logger.info("#### epochs: " + this.epochs );

		logger.info("#####################################################" );
	}

	@Override
	public void allocateClassifiers(AVTable data) {
		this.traindata = data;
		this.m = data.m;
		this.d = data.d;

		logger.info( "Num. of labels: " + this.m + " Dim: " + this.d );
		Random allocationRand = MasterSeed.nextRandom();

		logger.info( "Allocate the learners..." );

		this.w = new DenseVectorExt[this.m];
		this.thresholds = new double[this.m];
		this.stepfunctions = new StepFunction[this.m];
		for (int i = 0; i < this.m; i++) {
			this.w[i] = new DenseVectorExt(this.d + 1);
			this.stepfunctions[i] = this.stepFunction.clone();

			for (int j = 0; j <= d; j++)
				this.w[i].set(j, 2.0 * allocationRand.nextDouble() - 1.0);

			this.thresholds[i] = 0.2;
		}
		logger.info( "Done." );
	}

	protected void updatedPosteriors( int currIdx, int label, double inc ) {
		int n = traindata.x[currIdx].length;
		int[] indexes = new int[n + 1];
		double[] values = new double[n + 1];
		int indexx = 0;
		for (int feat = 0; feat < this.d; feat++) {
			if ((indexx < traindata.x[currIdx].length) &&
				(traindata.x[currIdx][indexx].index == feat)) {
				indexes[indexx] = feat;
				values[indexx] = inc * traindata.x[currIdx][indexx].value;
				indexx++;
			}
		}

		// Include bias term in weight vector:
		indexes[n] = this.d;
		values[n] = inc;

		SparseVectorExt grad = new SparseVectorExt(indexes, values, this.d + 1, n + 1);

		//grad.normalize();

		this.stepfunctions[label].step(this.w[label], grad);
	}

	protected ArrayList<Integer> shuffleIndex() {
		ArrayList<Integer> indirectIdx = new ArrayList<Integer>(this.traindata.n);
		for (int i = 0; i < this.traindata.n; i++) {
			indirectIdx.add(new Integer(i));
		}
		Collections.shuffle(indirectIdx, shuffleRand);
		return indirectIdx;
	}

	@Override
	public void train(AVTable data) {
		this.T = 1;

		for (int ep = 0; ep < this.epochs; ep++) {

			logger.info("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );

			ArrayList<Integer> indirectIdx = this.shuffleIndex();

			for (int i = 0; i < traindata.n; i++) {
				int currIdx = indirectIdx.get(i);

				int indexy = 0;
				for (int label = 0; label < traindata.m; label++) {
					double posterior = getPosteriors(traindata.x[currIdx], label);

					double currLabel = 0.0;
					if ((indexy < traindata.y[currIdx].length) && (traindata.y[currIdx][indexy] == label)) {
						currLabel = 1.0;
						indexy++;
					}

					// update the models
					double inc = posterior - currLabel;

					updatedPosteriors( currIdx, label, inc );

				}

				this.T++;

				if ((i % 10000) == 0) {
					logger.info( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ i +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					logger.info("\t\t" + dateFormat.format(date));
					logger.info("\t\t" + this.stepfunctions[0].toString() );
					//logger.info("\t\tWeight: " + this.w[0].get(0) );
					
				}

			}

			logger.info("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );


			//save model !!!!!!!!!!!!!!!!!!!!!!!
			//String modelFile = this.getProperties().getProperty("ModelFile");
			//this.savemodel(modelFile);
		}

	}



	@Override
	public double getPosteriors(AVPair[] x, int label) {
		double posterior = this.w[label].dot(x);

		posterior += this.w[label].get(this.d);
		posterior = s.value(posterior);
		return posterior;
	}

	public void save(String fname) {
		// TODO Auto-generated method stub
		try{
			logger.info( "Saving model (" + fname + ")..." );
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(fname)));

			for(int i = 0; i< this.w.length; i++ ){
				writer.write( ""+ this.w[i].get(this.d) );
				for(int j = 0; j< this.d; j++ ){
					writer.write( " "+ this.w[i].get(j) );
				}
				writer.write( "\n" );
			}

			writer.write( ""+ this.thresholds[0] );
			for(int i = 1; i< this.thresholds.length; i++ ){
				writer.write( " "+ this.thresholds[i] );
			}
			writer.write( "\n" );

			writer.close();
			logger.info( "Done." );
		} catch (IOException e) {
			logger.info(e.getMessage());
		}

	}

	public void load(String fname) {
		try {
			logger.info( "Loading model (" + fname + ")..." );
			Path p = Paths.get(fname);

			BufferedReader reader = Files.newBufferedReader(p, Charset.forName("UTF-8"));
		    String line = null;

		    // read file
		    ArrayList<String> lines = new ArrayList<String>();
		    while ((line = reader.readLine()) != null) {
		        lines.add(line);
		    }

		    reader.close();

		    // process lines
		    // allocate the model
		    this.m = lines.size()-1;
		    this.w = new DenseVectorExt[this.m];

		    double[][] weights = new double[this.m][];
		    for( int i = 0; i < this.m; i++ ){
		    	String[] values =  lines.get(i).split( " " );
		    	weights[i] = new double[values.length];
		    	weights[i][values.length - 1] = Double.parseDouble(values[0]);
		    	for( int j=1; j < values.length; j++ ){
		    		weights[i][j - 1] = Double.parseDouble(values[j]);
		    	}
		    }

		    this.d = weights[0].length - 1;
		    for (int i = 0; i < this.m; i++) {
		    	this.w[i] = new DenseVectorExt(this.d + 1);
		    this.w[i].set(this.d, weights[i][this.d]);
		    }

		    // last line for thresholds
		    this.thresholds = new double[this.m];
		    String[] values =  lines.get(lines.size()-1).split( " " );
	    	for( int j=0; j < values.length; j++ ){
	    		this.thresholds[j] = Double.parseDouble(values[j]);
	    	}



		    logger.info( "Done." );
		} catch (IOException x) {
		    System.err.format("IOException: %s%n", x);
		}

	}

}
