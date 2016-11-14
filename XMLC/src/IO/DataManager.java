package IO;

import java.io.InputStreamReader;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.Instance;

public abstract class DataManager {
	static Logger logger = LoggerFactory.getLogger(DataManager.class);
	
	public abstract boolean hasNext();
	public abstract Instance getNextInstance();
	public abstract int getNumberOfFeatures();
	public abstract int getNumberOfLabels();
	public abstract void setInputStream( InputStreamReader input );
	public abstract void reset();
	public abstract DataManager getCopy();
	
	public static DataManager managerFactory(String filename, String datamanagertype ) {
		DataManager datamanager = null;
				
		if (datamanagertype.compareTo( "Batch" ) == 0)
			datamanager = new BatchDataManager(filename);
		else if (datamanagertype.compareTo("Online") == 0)
			datamanager = new OnlineDataManager(filename);
		else {
			System.err.println("Unknown data manager");
			System.exit(-1);
		}
				
		return datamanager;		
		
	}
}
