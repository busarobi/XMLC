package threshold;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import IO.DataReader;
import IO.Evaluator;

public class MainThresholdTuningOutputThresholds extends MainThresholdTuning {
	private static Logger logger = LoggerFactory.getLogger(MainThresholdTuningOutputThresholds.class);
	
	private double[] thresholdsFTA = null;
	private double[] thresholdsEUM = null;
	private double[] thresholdsOFO = null;
	
	public MainThresholdTuningOutputThresholds(String fname) throws IOException {
		super(fname);
		// TODO Auto-generated constructor stub
	}

	protected void readOutFile() throws IOException{
		String[] fieldNames = {
			    "threshold",
			    "valid  Normalized macro F-measue (with m)",
			    "test  Normalized macro F-measue (with m)"
			};

		ArrayList<double[]> arrFTA = getNextArray(this.outFileName, fieldNames, "FTA" );
		ArrayList<double[]> arrEUM = getNextArray(this.outFileName, fieldNames, "EUM" );
		ArrayList<double[]> arrOFO = getNextArray(this.outFileName, fieldNames, "OFO" );


		System.out.println("########## FTA #######");
		this.printValues(arrFTA);
		int valFTAi = this.validateValues(arrFTA);
		this.thresholdsFTA = tuneThresholdFTA( valFTAi );
		
		System.out.println("########## EUM #######");
		this.printValues(arrEUM);
		int valEUMi = this.validateValues(arrEUM);
		this.thresholdsEUM = tuneThresholdEUM( valEUMi );
		
		System.out.println("########## OFO #######");
		this.printValues(arrOFO);
		int valOFOi = this.validateValues(arrOFO);
		this.thresholdsOFO = tuneThresholdOFO( valOFOi );
		
	}
	
	protected int validateValues( ArrayList<double[]> arr ) {
		int valIndex = 0;
		double valValue = 0.0;
		for( int i = 0; i < arr.size(); i++ ){
			if ( valValue < arr.get(i)[1] ) {
				valValue = arr.get(i)[1];
				valIndex = i;
			}
		}
		
		System.out.println("Validated: " + valIndex + ": " + arr.get(valIndex)[0] + "\t" + arr.get(valIndex)[1] + "\t" + arr.get(valIndex)[2] );
		
		return valIndex;
	}

	
	
	protected void loadPosteriors() throws Exception {
		
		DataReader validddatareader = new DataReader(this.lableFileValid, false, false);
		this.validlabels = validddatareader.read();
		this.validlabels.m = this.m;
		
		
		DataReader validpostreader = new DataReader(this.posteriorFileValid, false, false);
		this.validposteriors = validpostreader.read();
		this.validposteriors.m = this.m;

		if (this.fastxml){
			for( int i = 0; i < this.validposteriors.n; i++ ){
				for( int j = 0; j < this.validposteriors.x[i].length; j++ )
					this.validposteriors.x[i][j].index++;
			}
		}
		
		logger.info("Min. valid value : " + fmt(this.getMinimum(this.validposteriors)));
		
//		logger.info("Reading vaild psoteriors");
//		this.validposteriors = readPosteriors( this.posteriorFileValid, this.validlabels.n );
//		logger.info("Reading test psoteriors");
//		this.testposteriors = readPosteriors( this.posteriorFileTest, this.testlabels.n );
	}
	
	
	public double[] tuneThresholdOFO( int i) {		
		
		//for( int i=0; i < this.barray.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
		
			logger.info("Threshold: " + this.barray[i]);
			this.resultString += "OFO,threshold,"+this.barray[i] + "\n";
			
			//
			properties.setProperty("b", Integer.toString(this.barray[i]));
			
			
			ThresholdTuning tofo = new TTOfoFast( this.m, properties );
			double[] thresholds = tofo.validate(this.validlabels, this.validposteriors);
			
			this.resultString += "OFO,valid F-measure," + tofo.getValidatedFmeasure() + "\n";
			
			this.resultString += "OFO,valid num. of predicted positives," + tofo.getNumberOfPredictedPositives() + "\n";
			this.resultString += "OFO,valid avg. num. of predicted positives," + (tofo.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";
	
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### OFO valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "OFO,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			
			
		//}
		return thresholds;
	}
	
	
	public double[] tuneThresholdFTA( int i) {		
		//for( int i=0; i < this.thresholdForEUM.length; i++ ) {
			logger.info("##########################################################################");
			this.resultString += "##########################################################################\n";
			
			logger.info("Threshold: " + this.thresholdForEUM[i]);
			this.resultString += "FTA,threshold,"+this.thresholdForEUM[i] + "\n";
			
			// set the minThreshold
			//properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
			//ThresholdTuning theum = new TTEumFast( this.m, properties );
			double[] thresholds = new double[this.m];
			for( int j = 0; j < this.m; j++ ) thresholds[j] = this.thresholdForEUM[i];
			
			// compute the positive labels
			HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
			// compute F-measure
			Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

			for ( String perfName : perf.keySet() ) {
				logger.info("##### FTA valid " + perfName + ": "  + fmt(perf.get(perfName)));
				this.resultString += "FTA,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
			}
			
			

		//}
		return thresholds;
	}
	
	
	
	public double[] tuneThresholdEUM( int i ) {				
		logger.info("##########################################################################");
		this.resultString += "##########################################################################\n";
		
		logger.info("Threshold: " + this.thresholdForEUM[i]);
		this.resultString += "EUM,threshold,"+this.thresholdForEUM[i] + "\n";
		
		// set the minThreshold
		properties.setProperty("minThreshold", Double.toString(this.thresholdForEUM[i]));
		ThresholdTuning theum = new TTEumFast( this.m, properties );
		double[] thresholds = theum.validate(this.validlabels, this.validposteriors);
		
		this.resultString += "EUM,valid F-measure," + theum.getValidatedFmeasure() + "\n";
		
		this.resultString += "EUM,valid num. of predicted positives," + theum.getNumberOfPredictedPositives() + "\n";
		this.resultString += "EUM,valid avg. num. of predicted positives," + (theum.getNumberOfPredictedPositives()/ (double)this.validlabels.n) + "\n";

		// compute the positive labels
		HashSet<Integer>[] positiveLabelsArray = getPositiveLabels(this.validlabels, this.validposteriors, thresholds );
		// compute F-measure
		Map<String,Double> perf = Evaluator.computePerformanceMetrics(positiveLabelsArray, this.validlabels );

		for ( String perfName : perf.keySet() ) {
			logger.info("##### EUM valid " + perfName + ": "  + fmt(perf.get(perfName)));
			this.resultString += "EUM,valid " + perfName + ","  + fmt(perf.get(perfName)) + "\n";
		}
			
	    return thresholds;		
	}

	
	
	
	
	protected void printValues( ArrayList<double[]> arr ) {		
		for( int i = 0; i < arr.size(); i++ ){			
			for( int j=0; j < arr.get(i).length; j++ ){
				System.out.print(arr.get(i)[j] + "\t");
			}
			System.out.println();
		}
	}
	
	
	protected ArrayList<double[]> getNextArray( String OutFile, String[] fieldNames, String  method ) throws IOException {
		ArrayList<double[]> arr = new ArrayList<double[]>();
		
		BufferedReader fp = new BufferedReader(new FileReader(OutFile));
		
		for( int i = 0; i < 7; i++ ) {
			fp.readLine();
		}
		
		boolean readFlag = false;
		String line = null;
		int ii = 0;
		//arr.add(new double[fieldNames.length]);
		
		while ( true ) {
		    for( int i1 = 1; i1 <= 1000; i1++ ) {
		        line = fp.readLine();  
		        		        
		        if (line==null) break;
		        
		        //System.out.println(line);
		        if ((line.length() > 10 ) && (line.substring(0, 10).compareTo( "##########" )==0)) {
		            if (readFlag==true) {
		                ii = ii+1;		                
		                readFlag = false;
		            }            
		            break;
		        }
		        if (line.substring(0,method.length()).compareTo(method) != 0 ) {            
		            continue;
		        }else {
		            readFlag = true;
		        }
		        
		        for( int fi =0; fi < fieldNames.length; fi ++ ) {
		            String fn = method + "," + fieldNames[fi] + ",";
		            if (line.indexOf(fn) != -1 ) {
		                String nm = line.replace(fn, "");
		                
		                if (arr.size()<= ii) arr.add(new double[fieldNames.length]);
		                arr.get(ii)[fi] = Double.parseDouble(nm);
		                break;
		            }
		        }
		        
		    }
		    
		    if (line==null) break;
		}
		
		fp.close();

		//arr.remove(arr.size()-1);
		
		return arr;
	}
	
	
	
	public static void main(String[] args) throws Exception {
		logger.info("Working Directory = " + System.getProperty("user.dir"));


		// read properties
		if (args.length < 1) {
			logger.info("No config file given!");
			System.exit(-1);
		}
		
		MainThresholdTuningOutputThresholds th = new MainThresholdTuningOutputThresholds(args[0]);
		th.loadPosteriors();		
		th.readOutFile();
		
		
		

		th.writeOutThresholds();
	}

	protected void writeOutThresholds() throws IOException {
		String fname = this.outFileName.replaceAll(".txt" , "_thresholds.txt");
		BufferedWriter bf = new BufferedWriter(new FileWriter(fname) );
		
		for( int i= 0; i< this.thresholdsFTA.length; i++ ){		
			bf.write( i + "," + this.thresholdsFTA[i] + "," + this.thresholdsEUM[i] + "," + this.thresholdsOFO[i] + "\n" );
		}
		
		bf.close();
	}
	
	
	
}
