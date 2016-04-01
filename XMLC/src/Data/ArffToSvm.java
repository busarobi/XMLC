package Data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.TreeMap;
import java.util.TreeSet;

public class ArffToSvm {
    protected String inFile = null;
    protected String outFile = null;
    
	public ArffToSvm( String inFile, String outFile ) {
		this.inFile = inFile;
		this.outFile = outFile;
	}	
	
	public void convert() {
		String line = null;
		try {
			BufferedReader fp = new BufferedReader(new FileReader(this.inFile));

			int attrNum = 0;

			String classes = null;			
			
			while(true)
			{
				line = fp.readLine();
				if(line == null) {
					System.err.println("No data!!!");
					System.exit(-1);
				}
				
			    if ( ( line.length() >= 4 ) && (line.substring(0, 5).equalsIgnoreCase("@data"))  )
			        break;
			    else if ( line.contains("class") )
			        classes = line;			        
			    else if  ( ( line.length() >= 4 ) && ( line.substring(0, 5).equalsIgnoreCase("@attr") ) )			    
			        attrNum++;
			}
			
			System.out.println( "--> Attribute number: " + attrNum );
			
			int ind1 = classes.indexOf('{');
			int ind2 = classes.indexOf('}');
			classes = classes.substring(ind1+1, ind2-1);
			String[] classTokens = classes.split(",");
			
			TreeMap<String,String> labels = new TreeMap<String, String>();
			for(int i = 0; i < classTokens.length; i++ ) {
				labels.put(classTokens[i], Integer.toString(i+1) );
			}
			
			int lineNum = 0;
			BufferedWriter bf = new BufferedWriter(new FileWriter(this.outFile) );
			
			while(true)
			{
				line = fp.readLine();
				if(line == null) break;
				
				lineNum++;
				
				line = line.replace("{", "");
				line = line.replace("}", "");
				String[] tokens = line.split(",");				
				
				String[] labelTokens = tokens[tokens.length-1].trim().split(" ");
				
				String lab = labels.get(labelTokens[1]);
				if (lab==null) {					
					lab = Integer.toString(labels.size()+1);
					labels.put(labelTokens[1], lab);
					System.out.println("Missing label: " + lab);
					System.out.println(labelTokens[0] + " " + labelTokens[1] );
				}
				bf.write( lab );
				for( int i = 0; i < tokens.length-1; i++ ) {
					bf.write(" " + tokens[i].trim().replace(" ", ":" ) );
				}
				bf.write("\n");
				
				if(lineNum % 10000 == 0)
					System.out.println(lineNum);
				
			}
			
			System.out.println( "--> Line number: " + lineNum );
			
			fp.close();
			bf.close();
			
		} catch (Exception e) {
			System.out.println(line);
			e.printStackTrace();
		} 
	}
	
	public static void main(String[] args) {
		String inFile  = "/Users/busarobi/work/XMLC/data/mccdata/imageNet1K.arff";
		String outFile  = "/Users/busarobi/work/XMLC/data/mccdata/imageNet1K.txt";
		
		ArffToSvm af = new ArffToSvm( inFile, outFile );
		af.convert();
		
	}

}
