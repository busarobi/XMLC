package Learner;

import Data.AVTable;

public class AdamLR extends MLLogisitcRegression {

	protected double beta1;
	protected double beta2;
	protected double eps;



	public AdamLR(String propertyFileName) {
		super(propertyFileName);
		// Exponential decay rates for moment estimates in Adam
		this.beta1 = Double.parseDouble(this.properties.getProperty("beta1", "0.9"));
		this.beta2 = Double.parseDouble(this.properties.getProperty("beta2", "0.999"));
		this.gamma = Double.parseDouble(this.properties.getProperty("gamma", "0.001"));
		this.eps = Double.parseDouble(this.properties.getProperty("eps", "1e-8"));
	}
	
	

	@Override
	public void allocateClassifiers(AVTable data) {
		super.allocateClassifiers(data);
		System.out.print("Allocate the additional structures for Adam...");
		
		System.out.println("Done.");
	}



	@Override
	public void train(AVTable data) {
		// update the models
//		double inc = (currLabel - posterior);
//
//		int indexx = 0;
//		for (int l = 0; l < this.d; l++) {
//			if ((indexx < traindata.x[currIdx].length) && (traindata.x[currIdx][indexx].index == l)) {
//				this.grad[j][l] = inc * traindata.x[currIdx][indexx].value;
//				indexx++;
//			}
//		}
//
//		this.gradbias[j] = inc;
//		
//		// Adam moment updates
//		for (int l = 0; l < this.d; l++) {
//			this.firstmom[l] = this.beta1 * this.firstmom[l] + (1.0 - this.beta1) * this.grad[j][l];
//			this.secondmom[l] = this.beta2 * this.secondmom[l] + (1.0 - this.beta2) * Math.pow(this.grad[j][l], 2.0);
//			double bcFirst = this.firstmom[l] / (1.0 - Math.pow(this.beta1, this.T - 1.0));
//			double bcSecond = this.secondmom[l] / (1.0 - Math.pow(this.beta2, this.T - 1.0));
//			this.w[j][l] += this.step * bcFirst / (Math.sqrt(bcSecond) + this.eps);
//		}
//
//		// Why do we handle bias separately (?)
//		this.biasmom1 = this.beta1 * this.biasmom1 + (1.0 - this.beta1) * this.gradbias[j];
//		this.biasmom2 = this.beta2 * this.biasmom2 + (1.0 - this.beta2) * this.gradbias[j];
//		double bcFirst = this.biasmom1 / (1.0 - Math.pow(this.beta1, this.T - 1.0));
//		double bcSecond = this.biasmom2 / (1.0 - Math.pow(this.beta2, this.T - 1.0));
//		this.bias[j] += this.step * bcFirst / (Math.sqrt(bcSecond) + this.eps);
	}



	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println(Double.parseDouble("1e-8"));
	}

}
