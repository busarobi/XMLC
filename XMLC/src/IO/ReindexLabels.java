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
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.List;
import java.util.Map;
import jsat.linear.SparseVector;


import org.apache.commons.math3.util.Pair;

import Data.AVTable;
import Data.ComparablePair;


public class ReindexLabels {

	public static void main(String[] args) throws Exception {
		class ComparableIntegerPair implements Comparable<ComparableIntegerPair> {
		    private int  first;
		    private int second;

		    public ComparableIntegerPair( int key, int value ){
		    	this.first = key;
		    	this.second = value;
		    }
		    
		    public double getFirst() {
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
		}	
		
		
		
		String inputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped.csv";
		String outputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped_reindex.csv";
		
		DataReader datareader = new DataReader(inputFileName);
		AVTable data = datareader.read();

		BufferedWriter statbf = new BufferedWriter(new FileWriter("/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped_stats.txt") );		
		
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
		
		System.out.println("Number of different labels: " + labeltoindex.size() );

		
//		System.out.println("Allocating...");
//		
//		SparseVector cooccurence = new SparseVector(indextolabel.size()*indextolabel.size(), 10000000);
//		System.out.println("Done.");
//		
//		for(int i=0; i < data.n; i++) {
//			for( int j = 0; j < data.y[i].length; j++ ){
//				for( int k = j; k < data.y[i].length; k++ ){
//					int label1 = data.y[i][j];
//					int label2 = data.y[i][k];
//					
//					label1 = labeltoindex.get(label1);
//					label2 = labeltoindex.get(label2);
//					
//					if(label2 > label1 ) {
//						int tmp = label2;
//						label2 = label1;
//						label1 = tmp;
//					}
//					
//					int ind = (label1 * indextolabel.size()) + label2;
//					
//					Double itmp = cooccurence.get(ind);
//					if (itmp == 0.0) {
//						cooccurence.set(ind,1.0);
//					} else {
//						cooccurence.set(ind,itmp+1.0);
//					}
//					
//				}
//			}	
//
//			if ((i % 10000) == 0) {
//				System.out.println( "\t --> Instance: " + i + " (" + data.n + ")" );
//				DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
//				Date date = new Date();
//				System.out.println("\t\t" + dateFormat.format(date));
//				
//			}
//			
//		}
//		
//		
//		for( int i = 0; i < indextolabel.size(); i++ ){
//			int label1 = indextolabel.get(i); 
//			statbf.write( label1 );
//			
//			for( int j = 0; j < indextolabel.size(); j++ ){
//				int ind = (i * indextolabel.size()) + j;
//				double val = cooccurence.get(ind); 
//				if (val > 0.0){
//					statbf.write( "," + indextolabel.get(j) + "," + val  );
//				}
//			}
//			statbf.write( "\n");
//		}
//		
//		statbf.close();
//
//		
//		// re-indexing				
//		for(int i=0; i < data.n; i++) {
//			for( int j = 0; j < data.y[i].length; j++ ){				
//				int newlabel = labeltoindex.get(data.y[i][j]);
//				data.y[i][j] = newlabel;
//			}
//		}
//		
		BufferedReader fp = new BufferedReader(new FileReader(inputFileName));
		BufferedWriter bf = new BufferedWriter(new FileWriter(outputFileName) );
		
		for( int i = 0; i<data.n; i++)
		{
			// labels
			for(int j=0; j<data.y[i].length; j++ ) {
				//System.out.println(data.y[i][j]);
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
		}
		
		bf.close();
		fp.close();
		
	}

	
	
}
