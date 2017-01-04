package Learner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Properties;
import java.util.StringTokenizer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDoubleArray;

import Data.AVPair;
import Data.Instance;
import IO.DataManager;
import IO.ReadProperty;

public class DeepTreeLearner extends AbstractLearner {
	private static final long serialVersionUID = 1L;
	private static Logger logger = LoggerFactory.getLogger(DeepTreeLearner.class);
	
	protected int hiddendim = 0;
	protected String hiddenVectorsFile;
	protected String hiddenLabelVectorsFile;
	protected int numOfThreads = 1;
	
	protected DataManager traindata = null;
	protected double[][] hiddenWeights = null;		
	protected double[][] hiddenLabelRep = null;
	
	public DeepTreeLearner(Properties properties) {
		super(properties);
		System.out.println("#####################################################");
		System.out.println("#### Learner: DeepTreeLearner");

		this.hiddendim = Integer.parseInt(this.properties.getProperty("hiddendim", "100"));
		logger.info("#### Number of hidden dimension: " + this.hiddendim);

		this.hiddenVectorsFile = this.properties.getProperty("hiddenvectorsFile", null);
		logger.info("#### hidden vectors file name " + this.hiddenVectorsFile);

		this.hiddenLabelVectorsFile = this.properties.getProperty("hiddenlabelvectorsFile", null);
		logger.info("#### hidden label vectors file name " + this.hiddenLabelVectorsFile);
		
		//
		this.numOfThreads = Integer.parseInt(this.properties.getProperty("numThreads", "4"));
		logger.info("#### num of threads: " + this.numOfThreads);

		System.out.println("#####################################################");

	}

	@Override
	public void allocateClassifiers(DataManager data) {
		this.traindata = data;
		this.m = data.getNumberOfLabels();
		this.d = data.getNumberOfFeatures();

		this.readDeepRepresentationOfFeatures();
		this.buildLabelHiddenPresenation( data);
		this.writeHiddenLabelVectors(this.hiddenLabelVectorsFile);
	}

	protected void buildLabelHiddenPresenation( DataManager data) {
		logger.info("Computing the hidden label representations...");
		this.hiddenLabelRep = new double[this.m][];
		for (int i = 0; i < this.m; i++) {
			this.hiddenLabelRep[i] = new double[this.hiddendim];
		}

		data.reset();		
		int[] numelOfSums = new int[this.m];
		int ni = 0;
		while( data.hasNext() == true ){			
			Instance instance = data.getNextInstance();
			ni++;
			if (ni % 100000 == 0)
				logger.info("Processed label " + ni );
			
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
			for( int hdi = 0; hdi < this.hiddendim; hdi++ ){
				this.hiddenLabelRep[i][hdi] /= numelOfSums[i];
			}
		}
		
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
			BufferedReader fp = new BufferedReader(new FileReader(this.hiddenVectorsFile));
			
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
		// TODO Auto-generated method stub

	}

	@Override
	public double getPosteriors(AVPair[] x, int label) {
		// TODO Auto-generated method stub
		return 0;
	}

	public static void main(String[] args) {
		Properties properties = ReadProperty.readProperty("./examples/rcv1_traineval.config");
		
		DeepTreeLearner learner = new DeepTreeLearner(properties);
		
		DataManager traindata = DataManager.managerFactory(properties.getProperty("TrainFile"), "Online" );
		learner.allocateClassifiers(traindata);
		

	}

}
