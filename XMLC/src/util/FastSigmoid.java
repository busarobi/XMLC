package util;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FastSigmoid {
	private static Logger logger = LoggerFactory.getLogger(FastSigmoid.class);

	public static float fastSigmoid(float x) {
		return ((fastTanH(x) + 1)/(float)2.0);
	}
	
	public static float fastTanH(float x) {
	    if (x<0) return -fastTanH(-x);
	    if (x>8) return 1f;
	    float xp = TANH_FRAC_BIAS + x;
	    short ind = (short) Float.floatToRawIntBits(xp);
	    float tanha = TANH_TAB[ind];
	    float b = xp - TANH_FRAC_BIAS;
	    x -= b;
	    return tanha + x * (1f - tanha*tanha);
	}

	private static final int TANH_FRAC_EXP = 6; // LUT precision == 2 ** -6 == 1/64
	private static final int TANH_LUT_SIZE = (1 << TANH_FRAC_EXP) * 8 + 1;
	private static final float TANH_FRAC_BIAS =
	    Float.intBitsToFloat((0x96 - TANH_FRAC_EXP) << 23);
	private static float[] TANH_TAB = new float[TANH_LUT_SIZE];
	static {
	    for (int i = 0; i < TANH_LUT_SIZE; ++ i) {
	        TANH_TAB[i] = (float) Math.tanh(i / 64.0); 
	    }
	}	
	
	public static void main(String[] args) {
		Random r = new Random();
		int rep = 1000000000;
		
		
		logger.info("Number of calls: " + rep );
		DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
		Date date = new Date();
		logger.info("Fast sigmoid starts\t\t" + dateFormat.format(date));

		for( int i = 0; i < rep; i++ ){
			double val = (r.nextDouble()-0.5) * 100.0;
			val = FastSigmoid.fastSigmoid((float)val);
		}

		
		date = new Date();
		logger.info("Fast sigmoid ends\t\t" + dateFormat.format(date));

		
		Sigmoid s = new Sigmoid();


		date = new Date();
		logger.info("Sigmoid starts\t\t" + dateFormat.format(date));

		for( int i = 0; i < rep; i++ ){
			double val = (r.nextDouble()-0.5) * 100.0;
			val = s.value(val);
		}

		
		date = new Date();
		logger.info("Sigmoid ends\t\t" + dateFormat.format(date));
		
		
	}

}
