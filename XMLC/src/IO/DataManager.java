package IO;

import java.io.InputStreamReader;

import Data.Instance;

public interface DataManager {
	public boolean hasNext();
	public Instance getNextInstance();
	public int getNumberOfFeatures();
	public int getNumberOfLabels();
	public void setInputStream( InputStreamReader input );
	public void reset();
	public DataManager getCopy();
}
