package IO;

import java.io.InputStreamReader;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import Data.Instance;

public interface DataManager {
	static Logger logger = LoggerFactory.getLogger(DataManager.class);
	
	public boolean hasNext();
	public Instance getNextInstance();
	public int getNumberOfFeatures();
	public int getNumberOfLabels();
	public void setInputStream( InputStreamReader input );
	public void reset();
	public DataManager getCopy();
	
	static DataManager managerFactory(String filename, String datamanagertype ) {
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
