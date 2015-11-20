package Learner;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;

import Data.AVTable;

public class AdamLR extends MLLogisticRegression {

	protected double[][] firstmom = null;
	protected double[][] secondmom = null;
	protected double[] biasfm = null;
	protected double[] biassm = null;

	protected double beta1;
	protected double beta2;
	protected double eps;
	protected double lambda;



	public AdamLR(Properties properties) {
		super(properties);
		System.out.println("#####################################################");
		System.out.println("#### Optimizer: Adam");
		// Exponential decay rates for moment estimates in Adam
		this.beta1 = Double.parseDouble(this.properties.getProperty("beta1", "0.9"));
		System.out.println("#### beta1: " + this.beta1);
		this.beta2 = Double.parseDouble(this.properties.getProperty("beta2", "0.999"));
		System.out.println("#### beta2: " + this.beta2);
		this.eps = Double.parseDouble(this.properties.getProperty("eps", "1e-8"));
		System.out.println("#### eps: " + this.eps);
		this.lambda = Double.parseDouble(this.properties.getProperty("lambda", "0.0"));
		System.out.println("#### lambda: " + this.lambda);
		System.out.println("#####################################################" );
	}



	@Override
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);
		System.out.print("Allocate additional structures for Adam...");

		this.grad = new double[this.m][];
		this.gradbias = new double[this.m];
		this.firstmom = new double[this.m][];
		this.secondmom = new double[this.m][];
		this.biasfm = new double[this.m];
		this.biassm = new double[this.m];
		for (int i = 0; i < this.m; i++) {
			this.grad[i] = new double[d];
			this.firstmom[i] = new double[d];
			this.secondmom[i] = new double[d];
		}

		System.out.println("Done.");
	}



	@Override
	public void train(AVTable data) {
		this.T = 1;
		for (int ep = 0; ep < this.epochs; ep++) {
			System.out.println("#############--> BEGIN of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
			ArrayList<Integer> indirectIdx = this.shuffleIndex();
			for (int row : indirectIdx) {

				int label_index = 0;
				double learningRate = this.gamma * Math.sqrt(1.0 - Math.pow(this.beta2, this.T))
		                   / (1.0 - Math.pow(this.beta1, this.T));
				for (int label = 0; label < traindata.m; label++) {
					double posterior = getPosteriors(traindata.x[row], label);
					double currentLabel = 0.0;
					if ((label_index < traindata.y[row].length) &&
						(traindata.y[row][label_index] == label)) {
						currentLabel = 1.0;
						label_index++;
					}
					double inc = posterior - currentLabel;

					int indexx = 0;
					for (int feat = 0; feat < this.d; feat++) {
						if ((indexx < traindata.x[row].length) &&
							(traindata.x[row][indexx].index == feat)) {
							this.grad[label][feat] = inc * traindata.x[row][indexx].value
								+ this.lambda * this.w[label][feat];
							indexx++;
						}


						this.firstmom[label][feat] =
							this.beta1 * this.firstmom[label][feat]
							+ (1.0 - this.beta1) * this.grad[label][feat];
						this.secondmom[label][feat] =
							this.beta2 * this.secondmom[label][feat]
							+ (1.0 - this.beta2) * Math.pow(this.grad[label][feat], 2.0);
						this.w[label][feat] -= learningRate * this.firstmom[label][feat]
							/ (Math.sqrt(this.secondmom[label][feat]) + this.eps);
					}
					this.gradbias[label] = inc + this.lambda * this.bias[label];
					this.biasfm[label] = this.beta1 * this.biasfm[label] + (1.0 - this.beta1) * this.gradbias[label];
					this.biassm[label] = this.beta2 * this.biassm[label] + (1.0 - this.beta2) * Math.pow(this.gradbias[label], 2.0);
					this.bias[label] -= learningRate * this.biasfm[label] / (Math.sqrt(this.biassm[label]) + this.eps);
				}
				this.T++;

				if (this.T % 1000 == 0) {
					System.out.println( "\t --> Epoch: " + (ep+1) + " (" + this.epochs + ")" + "\tSample: "+ this.T +" (" + data.n + ")" );
					DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
					Date date = new Date();
					System.out.println("\t\t" + dateFormat.format(date));
					System.out.println("Weight: " + this.w[0][0] );
				}
			}
			System.out.println("--> END of Epoch: " + (ep + 1) + " (" + this.epochs + ")" );
		}
	}



	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println(Double.parseDouble("1e-8"));
	}

}
