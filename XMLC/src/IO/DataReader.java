package IO;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.Vector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVPair;
import Data.AVTable;

public class DataReader {
	private static Logger logger = LoggerFactory.getLogger(DataReader.class);

	protected boolean additionalStat = false;
	protected boolean initialline = false;
	
	protected String fileName = null;
	
	public DataReader( String fileName )
	{
		this.fileName = fileName;
	}

	public DataReader( String fileName, boolean additionalStatistics )
	{
		this.fileName = fileName;
		this.additionalStat = additionalStatistics;
	}
	
	public DataReader( String fileName, boolean additionalStatistics, boolean initline )
	{
		this.fileName = fileName;
		this.additionalStat = additionalStatistics;
		this.initialline = initline;
	}

	public AVTable read() throws Exception
	{
		logger.info( "Reading " + this.fileName + "..." );
		
		BufferedReader fp = new BufferedReader(new FileReader(this.fileName));
		Vector<int[]> vy = new Vector<int[]>();
		Vector<AVPair[]> vx = new Vector<AVPair[]>();
		int max_feature_index = 0;
		int max_label_index = 0;

		// for additional statistics
		HashSet<Integer> hashsetFeatures = new HashSet<Integer>(); 
		HashSet<Integer> hashsetLabels = new HashSet<Integer>(); 
		//
		int ni = 0, di = 0, mi = 0;		
		if (this.initialline){
			String line = fp.readLine();
			String[] tokens = line.split(" ");
						
			ni = Integer.parseInt(tokens[0]);
			di = Integer.parseInt(tokens[1]);
			mi = Integer.parseInt(tokens[2]);
		}
		
		
		while(true)
		{
			String line = fp.readLine();
			if(line == null) break;
			
			line = line.replace(",", " " );
			StringTokenizer st = new StringTokenizer(line," \t\n\r\f");			
			int m = st.countTokens();
			String[] stArr = new String[m];
			
			int numOfFeatures = 0;
			int numOfLabels = 0;
			for(int j=0;j<m;j++)
			{
				stArr[j] = st.nextToken();
				if (stArr[j].contains(":") ) numOfFeatures++;
				else numOfLabels++;
			}
			
			
												
			AVPair[] x = new AVPair[numOfFeatures];
			int[] y = new int[numOfLabels];
			int indexx = 0;
			int indexy = 0;
			for(int j=0;j<m;j++)
			{
				String[] tokens = stArr[j].split(":");
				//logger.info( stArr[j] );
				if ( tokens.length == 1 ) {  // label
					//logger.info( tokens[0].replace(",", "") );					
					y[indexy]= Integer.parseInt(tokens[0].replace(",", ""));
					if (additionalStat)
						hashsetLabels.add(y[indexy]);
					indexy++;
				} else {  // features
					x[indexx] = new AVPair();
					x[indexx].index = Integer.parseInt(tokens[0])-1;         // the indexing starts at 0
					if (additionalStat) 
						hashsetFeatures.add(x[indexx].index);
					x[indexx++].value = Double.parseDouble(tokens[1]);							
				}
			}
			
			Arrays.sort(y);
			
			if(indexx>0) max_feature_index = Math.max(max_feature_index, x[indexx-1].index);
			if(indexy>0) max_label_index = Math.max(max_label_index, y[indexy-1]);
									
			vy.addElement(y);
			vx.addElement(x);
		}
		
		fp.close();
		
		AVTable data = new AVTable();
		data.n = vy.size();
		data.x = new AVPair[data.n][];
		data.d = max_feature_index+1;
		data.m = max_label_index+1;
		
		if (this.initialline){
			if (data.n != ni) { 
				logger.info("Number of line does not match with data given in the header");
				System.exit(-1);
			}
			data.d = di;
			data.m = mi;					
		}
		
		for(int i=0;i<data.n;i++)
			data.x[i] = vx.elementAt(i);
		data.y = new int[data.n][];
		for(int i=0;i<data.n;i++)
			data.y[i] = vy.elementAt(i);
		
		logger.info( "Done." );

		logger.info( "    -->  num x dim: labels " 
		           + data.n + " x "+ data.d + " : " + data.m );
		
		
		if ( additionalStat ) {
			logger.info("    -->  Number of distinct features: " + hashsetFeatures.size());
			logger.info("    -->  Number of distinct labels: " + hashsetLabels.size());
			
			
			HashSet<Pair> hashsetPairs = new HashSet<Pair>();
			
			for( int i = 0; i<data.n; i++) 	{
				for(int m=0; m<data.y[i].length; m++ ) {
					for(int j=0; j<data.x[i].length; j++ ) {
						Pair pair = new Pair(data.y[i][m], data.x[i][j].index);
						hashsetPairs.add(pair);
					}				
				}
				
			}
	
			logger.info("    -->  Number of distinct pairs: " + hashsetPairs.size());
		}
		
		return data;
	}

		
	class Pair {
			
		int label = 0;
		int feature = 0;
			
		public Pair(int label, int feature) {
			this.label = label;
			this.feature = feature;
		}
					
		public boolean equals(Object obj) {
		
			if(obj != null && obj instanceof Pair) {
		      	Pair pair = (Pair) obj;
		        if(this.label != pair.label) return false;
		        if(this.feature != pair.feature) return false;
		        return true;
	        }
	        return false;
	    }
			
		public int hashCode() {
	        int hash = 1;
	        hash = hash * 17 + label;
	        hash = hash * 31 + feature;
	        return hash;
	    }
			
	}

		
	public void write( String outfname, AVTable data ) throws IOException
	{
		BufferedWriter bf = new BufferedWriter(new FileWriter(outfname) );
		
		for( int i = 0; i<data.n; i++)
		{
			// labels
			for(int j=0; j<data.y[i].length; j++ ) {
				//logger.info(data.y[i][j]);
				bf.write(  "" + data.y[i][j]  );
				if ( j < data.y[i].length - 1 )
					bf.write( "," );
				else
					bf.write( " " );
			}		
			// features
			for(int j=0; j<data.x[i].length; j++ ) {
				//bf.write(  (data.x[i][j].index + 1) + ":" + String.format( "%g", data.x[i][j].value) );
				bf.write(  (data.x[i][j].index + 1) + ":" + fmt(data.x[i][j].value) );
				if ( j < data.x[i].length - 1 )
					bf.write( " " );
			}				
			
			bf.write( "\n" );
		}
		
		bf.close();
	}

	public static void writeLabels( String outfname, AVTable data ) throws IOException
	{
		BufferedWriter bf = new BufferedWriter(new FileWriter(outfname) );
		
		for( int i = 0; i<data.n; i++)
		{
			// labels
			for(int j=0; j<data.y[i].length; j++ ) {
				//logger.info(data.y[i][j]);
				bf.write(  "" + data.y[i][j]  );
				if ( j < data.y[i].length - 1 )
					bf.write( "," );
				else
					bf.write( "\n" );
			}		
		}
		
		bf.close();
	}
	
	
	
	public String fmt(double d)
	{
	    if(d == (long) d)
	        return String.format("%d",(long)d);
	    else
	        return String.format("%s",d);
	}

	
	
	public static void main(String[] args) throws Exception {
		// String fileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train_small.csv";
		// String outfileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train_small_2.csv";
		
		String fileName = "/Users/busarobi/work/XMLC/data/scene/scene_train";
		String outfileName = "/Users/busarobi/work/XMLC/data/scene/scene_train_2";
		
		DataReader datareader = new DataReader( fileName );
		AVTable data = datareader.read();
		
		logger.info( "N: " + data.n + " D: " + data.d + " M: " + data.m );
		
		datareader.write(outfileName, data);		
	}

}
