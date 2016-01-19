package IO;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;


public class ReindexLabels {
	private static Logger logger = LoggerFactory.getLogger(ReindexLabels.class);

	public static void main(String[] args) throws Exception {
		
		class ComparableIntegerPair implements Comparable<ComparableIntegerPair>  {
		    private int  first;
		    private int second;

		    public ComparableIntegerPair( int key, int value ){
		    	this.first = key;
		    	this.second = value;
		    }
		    
		    public int getFirst() {
		        return this.first;
		    }

		    public int getSecond() {
		        return this.second;
		    }

		    @Override
			public int compareTo(ComparableIntegerPair o) {
				if (this.first == o.first) {
					if (this.second == o.second) return 0;
					else {
						if ( this.second < o.second ) return -1;
						else return 1;
					}
				} else if (this.first<o.first){
					return -1;
				} else return 1;
			}
			
			@Override
		    public boolean equals(Object o) {
		        //if (this == o) return true;
		        //if (!(o instanceof ComparableIntegerPair)) return false;
		        ComparableIntegerPair key = (ComparableIntegerPair) o;
		        return first == key.first && second == key.second;
		    }

		    @Override
		    public int hashCode() {
		    	return new Integer(first).hashCode() * 31 + new Integer(second).hashCode();
		    }

			
			
		}	
		
		
		
		//String inputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped.csv";
		String inputFileName = "/Users/busarobi/work/XMLC/data/Amazon/amazon_train.txt";
		
		DataReader datareader = new DataReader(inputFileName, true, true);
		AVTable data = datareader.read();

				
		
		HashMap<Integer,Integer> labeltoindex = new HashMap<>();
		ArrayList<Integer> indextolabel = new ArrayList<>();
		
		int index = 0;
		for(int i=0; i < data.n; i++) {
			for( int j = 0; j < data.y[i].length; j++ ){
				if ( ! labeltoindex.containsKey(data.y[i][j] ) )
					labeltoindex.put(data.y[i][j], index++);
					indextolabel.add(data.y[i][j]);
			}
		}
		
		logger.info("Number of different labels: " + labeltoindex.size() );

		
		logger.info("Allocating...");
		
		Map<ComparableIntegerPair, Integer> cooccurence = new TreeMap<ComparableIntegerPair, Integer>();
		
		logger.info("Done.");
		
		for(int i=0; i < data.n; i++) {
			for( int j = 0; j < data.y[i].length; j++ ){
				for( int k = j; k < data.y[i].length; k++ ){
					int label1 = data.y[i][j];
					int label2 = data.y[i][k];
					
					//label1 = labeltoindex.get(label1);
					//label2 = labeltoindex.get(label2);
					
					ComparableIntegerPair cp = null;
					if(label2 > label1 ) 
						cp = new ComparableIntegerPair(label1, label2);
					else
						cp = new ComparableIntegerPair(label2, label1);
					
					
					Integer itmp = cooccurence.get(cp);
					if (itmp == null) {
						cooccurence.put(cp,1);
					} else {
						cooccurence.remove(cp);
						itmp++;
						cooccurence.put(cp,itmp);
					}
					
				}
			}	

			if ((i % 10000) == 0) {
				logger.info( "\t --> Instance: " + i + " (" + data.n + ")" );
				logger.info("\t\t Size: " + cooccurence.size() );
				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
				Date date = new Date();
				logger.info("\t\t" + dateFormat.format(date));
				
			}
			
		}
		
		logger.info("Writing the stat...");
		
		
		BufferedWriter statbf = new BufferedWriter(new FileWriter("/Users/busarobi/work/XMLC/data/Amazon/train-remapped_stats.txt") );
		
		for( ComparableIntegerPair cp : cooccurence.keySet() ) {
			statbf.write( "" + cp.getFirst() + "," + cp.getSecond() + "," + cooccurence.get(cp) +"\n" );
		}
		
		
//		for( int i = 0; i < indextolabel.size(); i++ ){
//			int label1 = indextolabel.get(i); 
//			statbf.write( "" + label1 );
//			
//			for( int j = 0; j < indextolabel.size(); j++ ){				
//				int label2 = indextolabel.get(j);
//				
//				ComparableIntegerPair cp = null;
//				if(label2 > label1 ) 
//					cp = new ComparableIntegerPair(label1, label2);
//				else
//					cp = new ComparableIntegerPair(label2, label1);
//				
//				
//				Integer val = cooccurence.get(cp); 
//				if (val != null){
//					statbf.write( "," + label2 + "," + val  );
//				}
//			}
//			statbf.write( "\n");
//			statbf.flush();
//		}
		
		statbf.close();

		logger.info("Done.");
		
		
		// re-indexing				
		for(int i=0; i < data.n; i++) {
			for( int j = 0; j < data.y[i].length; j++ ){				
				int newlabel = labeltoindex.get(data.y[i][j]);
				data.y[i][j] = newlabel;
			}
		}
		
		
		logger.info("Writing out the dataset...");
		
		BufferedReader fp = new BufferedReader(new FileReader(inputFileName));
		
		String outputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped_reindex.csv";
		BufferedWriter bf = new BufferedWriter(new FileWriter(outputFileName) );
		
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
			String line = fp.readLine();
			String[] tokens = line.split( " " );
			for( int j=data.y[i].length; j < tokens.length; j++ ){
				bf.write( tokens[j] );
				if ( j<tokens.length-1){
					bf.write( " " );
				}
			}
			
			
			bf.write( "\n" );
			
			if ((i % 10000) == 0) {
				logger.info("Line: " + i + "(" + data.n + ")" );
			}
			
		}
		
		bf.close();
		fp.close();
		
	}

	
	
}
