package IO;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Properties;

import Data.AVTable;
import Data.Instance;

public class BatchDataManager extends DataManager {
	protected boolean isHeader = true;
	protected AVTable data = null;
	protected int currentIndex = 0;
	
	public BatchDataManager( String filename ) {
		DataReader datareader = new DataReader(filename, false, this.isHeader);
		try {
			data = datareader.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.currentIndex = 0;
	}
	
	public BatchDataManager( AVTable data ) {
		this.data = data;
		this.currentIndex = 0;
	}

	@Override
	public boolean hasNext() {
		if (currentIndex >= data.n )
			return false;
		else
			return true;
	}

	@Override
	public Instance getNextInstance() {
		Instance instance = null;
		if (currentIndex < data.n){
			instance = new Instance(data.x[currentIndex], data.y[currentIndex]);
			currentIndex++;
		}
		return instance;
	}

	@Override
	public int getNumberOfFeatures() {
		return data.d;
	}

	@Override
	public int getNumberOfLabels() {
		return data.m;
	}
	
	@Override
	public void setInputStream( InputStreamReader input ) {
		
	}

	@Override
	public void reset() {
		this.currentIndex = 0;
		
	}

	@Override
	public DataManager getCopy() {
		return new BatchDataManager(this.data);		
	}
}
