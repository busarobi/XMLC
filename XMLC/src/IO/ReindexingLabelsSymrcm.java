package IO;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.AVTable;

public class ReindexingLabelsSymrcm {
	private static Logger logger = LoggerFactory.getLogger(ReindexingLabelsSymrcm.class);


	public static void main(String[] args) throws Exception {
		
		
		
		//read permutation
		logger.info("Reading permuation...");
		//String inputPermFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/permutation.txt";
		String inputPermFileName = "./permutation_sym_tree.txt";
		BufferedReader fperm = new BufferedReader(new FileReader(inputPermFileName));
		HashMap<Integer,Integer> mapping = new HashMap<>();
		
		while(true)
		{
			String line = fperm.readLine();
			if(line == null) break;
			
			String[] tokens = line.split(",");
			
			mapping.put(Integer.parseInt(tokens[0]), Integer.parseInt(tokens[1]));
		}		
				
		fperm.close();
		
		logger.info("Done.");
		
		//String inputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped.csv";
		//String inputFileName = "./train-remapped.csv";
		String inputFileName = args[0];
		DataReader datareader = new DataReader(inputFileName);
		AVTable data = datareader.read();

				
		
		
		
		
		
		// re-indexing				
		for(int i=0; i < data.n; i++) {
			for( int j = 0; j < data.y[i].length; j++ ){				
				int newlabel = mapping.get(data.y[i][j]);
				data.y[i][j] = newlabel;
			}
		}
		
		
		logger.info("Writing out the dataset...");
		
		BufferedReader fp = new BufferedReader(new FileReader(inputFileName));
		
		//String outputFileName = "/Users/busarobi/work/Fmeasure/LSHTC/dataraw/train-remapped_reindex_symrcm.csv";
		//String outputFileName = "./train-remapped_reindex_symrcm.csv";
		String outputFileName = args[1];
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
				logger.info("Line: " + i + " (" + data.n + ")" );
			}
			
		}
		
		bf.close();
		fp.close();
	

	}

}
