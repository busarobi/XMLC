package Learner.step;

import java.util.Properties;

import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.lossfunctions.LogisticLoss;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.PowerDecay;

public class GradStepL1 implements StepFunction {
	//private LossFunc loss = new LogisticLoss();;
    //private GradientUpdater gradientUpdater;
    private double eta;
    private DecayRate decay = new PowerDecay(1, 0.1);
    private int time;
    private double lambda0;
    private double lambda1;
    private double l1U;
    private double[] l1Q;    

    public GradStepL1() {    
    }
    
    
	public GradStepL1(Properties properties) {
		System.out.println("#####################################################");
		System.out.println("#### Optimizer: Simple gradient descent with L1 penalization");
		
		this.eta = Double.parseDouble(properties.getProperty("eta", "0.001"));
		System.out.println("#### gamma: " + this.eta);

		this.lambda0 = Double.parseDouble(properties.getProperty("lambda0", "0.001"));
		System.out.println("#### lambda0: " + this.lambda0 );
		
		this.lambda1 = Double.parseDouble(properties.getProperty("lambda1", "0.001"));
		System.out.println("#### lambda1: " + this.lambda1 );
		
		
		
		System.out.println("#####################################################");
	}

	private void allocate(int length) {
        time = 0;
        l1U = 0;
        if(lambda1 > 0)
            l1Q = new double[length];
        else
            l1Q = null;
	}
	
	

	@Override
	public void step(Vec w, SparseVector grad ) {
		if (time <= 0) {
			allocate(w.length());
		}
		
        final double eta_t = decay.rate(time++, eta);
        //Vec x = dataPoint.getNumericalValues();
        
        
        //applyL2Reg(eta_t);
        if(lambda0 > 0)//apply L2 regularization
            w.mutableMultiply(1-eta_t*lambda0);
        
        w.mutableSubtract( eta_t, grad );
        
        //applyL1Reg(eta_t, x);
        if(lambda1 > 0)
        {
            l1U += eta_t*lambda1;//line 6: in Tsuruoka et al paper, figure 2
            for(IndexValue iv : grad)
            {
                final int i = iv.getIndex();
                //see "APPLYPENALTY(i)" on line 15: from Figure 2 in Tsuruoka et al paper
                final double z = w.get(i);
                double newW_i;
                if (z > 0)
                    newW_i = Math.max(0, z - (l1U + l1Q[i]));
                else
                    newW_i = Math.min(0, z + (l1U - l1Q[i]));
                l1Q[i] += (newW_i - z);
                w.set(i, newW_i);                
            }
        }
        
    }
			

	@Override
	public double step(Vec w, SparseVector grad, double bias, double biasGrad) {
		return 0.0;
	}

	@Override
	public StepFunction clone() {
		GradStepL1 newstep = new GradStepL1();
		
		newstep.eta = this.eta;
		newstep.lambda0 = this.lambda0;
		newstep.lambda1 = this.lambda1;
		
		
		return (StepFunction)newstep;
	}

	public String toString() {
		String r = "";
		r += "Decay: " + decay.rate(time, eta);
		return r;
	}

	public double getEta() {
		return eta;
	}

	public DecayRate getDecay() {
		return decay;
	}

	public double getLambda0() {
		return lambda0;
	}

	public double getLambda1() {
		return lambda1;
	}

	public double getL1U() {
		return l1U;
	}
	
	
	
}
