package Data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SyntheticData {
	private static Logger logger = LoggerFactory.getLogger(SyntheticData.class);

	
	public static double getPosterior(double[] w, double bias, double [] x) {
		
		double posteriors = bias;
		
		for(int i = 0; i < w.length; i++) {
			posteriors += w[i] * x[i];
		}
		
		return  s.value(posteriors);	
	}
	
	public static double[] generateExample(Random random, int d) {
	
		double [] x = new double[d];
		for(int i = 0; i < d; i++) {
			x[i] = 4*random.nextDouble() - 2;
		}
		return x;
	}
	
	public static String exampleToString(double [] x) {
		String str = "";
		for(int i = 0; i < x.length; i++) {
			if(i < x.length - 1) {
				str += (i + ":" + x[i] + " ");
			} else {
				str += (i + ":" + x[i]);
			}
		}
		return str;
	}
	
	public static void generateData(double [][] w, double [] bias, Random random, int n, int d, int m, String filename) throws Exception {
		
		BufferedWriter writer = new BufferedWriter(new FileWriter((filename)));
		
		for(int i = 0; i < n; i++) {
			double[] x = generateExample(random, d);
			boolean first = true;
			for(int j = 0; j < m; j++) {
				double p = getPosterior(w[j], bias[j], x);
				int y = random.nextDouble() < p ? 1 : 0;
				if(y == 1) {
					if(first) {
						logger.info(Integer.toString(j));
						writer.write("" + j);
						first = false;
					}
					else {
						logger.info(","+j);
						writer.write(","+j);
					}
				}
			}
			String str = exampleToString(x);
			logger.info(" " + str);
			writer.write(" " + str + "\n");			
			
		}
		
		writer.close();
	}
	
	static Sigmoid s = new Sigmoid();
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		int m = 1024;
		int d = 16;
		int n = 10000;
		
		double [][] w = new double[m][d];
		double [] bias = new double[m];
		
		Random random = new Random(1);
		
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < d; j++) {
				w[i][j] = (2*random.nextDouble() - 1.0);
			}
			bias[i] = random.nextDouble() - 7.0;
		}
		
		generateData(w, bias, random, n, d, m, "../data/synthetic_data/train.svm");
		generateData(w, bias, random, n, d, m, "../data/synthetic_data/validate.svm");
		generateData(w, bias, random, n, d, m, "../data/synthetic_data/test.svm");
		
		
	}

}
