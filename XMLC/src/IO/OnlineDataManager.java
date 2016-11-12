package IO;

import java.io.InputStreamReader;

import Data.Instance;

public class OnlineDataManager implements DataManager {

	public OnlineDataManager() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Instance getNextInstance() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getNumberOfFeatures() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getNumberOfLabels() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setInputStream(InputStreamReader input) {
		// TODO Auto-generated method stub

	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub

	}

	@Override
	public DataManager getCopy() {
		// TODO Auto-generated method stub
		return null;
	}

}
