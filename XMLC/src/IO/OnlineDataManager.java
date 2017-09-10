package IO;

import Data.AVPair;
import Data.Instance;

import java.io.*;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class OnlineDataManager extends DataManager {
	protected ReaderThread  readerthread = null;
	protected String filename = null;
	protected BlockingQueue<Instance> blockingQueue = null;
	protected int bufferSize = 16384;
	protected Thread rthread = null;
	protected int processedItem = 0;
	protected int nFeatures = 0;
	protected int nLabels = 0;
	
	public OnlineDataManager(String filename) {
		this.filename = filename;		
		this.blockingQueue = new ArrayBlockingQueue<Instance>(this.bufferSize);
		this.readerthread = new ReaderThread(this.blockingQueue, this.filename);

		this.nFeatures = this.readerthread.d;
		this.nLabels = this.readerthread.m;
		
		this.rthread = new Thread(this.readerthread);
		this.rthread.start();		
	}

	@Override
	public boolean hasNext() {
		if (this.blockingQueue.size() > 0) return true;
		if ((! this.readerthread.isEndOfFile() )) { // not end of line, but we do not know whether new instance will come
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			if (this.blockingQueue.size() > 0) return true;
			if (this.readerthread.isEndOfFile()) return false;
		}
		return false;
	}

	@Override
	public Instance getNextInstance() {
		Instance instance = null;
		try {
			instance = this.blockingQueue.take();
			this.processedItem++;
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return instance;		
	}

	@Override
	public int getNumberOfFeatures() {
		return this.nFeatures;
	}

	@Override
	public int getNumberOfLabels() {		
		return this.nLabels;
	}

	@Override
	public void setInputStream(InputStreamReader input) {
		// TODO Auto-generated method stub

	}

	@Override
	public void reset() {
		this.close();
		this.blockingQueue = new ArrayBlockingQueue<Instance>(this.bufferSize);
		this.readerthread = new ReaderThread(this.blockingQueue, this.filename);
		this.rthread = new Thread(this.readerthread);
		this.rthread.start();				
	}

	@Override
	public DataManager getCopy() {
		return new OnlineDataManager(this.filename);
	}

	// for reading data
	public class ReaderThread implements Runnable{

		  protected BlockingQueue<Instance> blockingQueue = null;
		  //protected final Semaphore available = new Semaphore(1);
		  protected String filename = null;
		  protected int d; 
		  protected int n;
		  protected int m;
		  
		  protected volatile boolean endOfFile = false;
		  public volatile boolean flag = true;
		  
		  BufferedReader br = null;
		  
		  public ReaderThread(BlockingQueue<Instance> blockingQueue, String filename){
		    this.blockingQueue = blockingQueue;
		    this.filename = filename;
		    
		    try {
				this.br = new BufferedReader(new FileReader(new File(this.filename)));
				String line = this.br.readLine();
				// process line
				String[] tokens = line.split(" ");
				
				this.n = Integer.parseInt(tokens[0]);
				this.d = Integer.parseInt(tokens[1]);
				this.m = Integer.parseInt(tokens[2]);				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		    
		  }

		  @Override
		  public void run() {
		     try {
		            String buffer = br.readLine();
		            if (buffer == null ) {
		            	this.endOfFile = true;
		            }

	            	if (this.endOfFile == false ) {     			            
			            while(true){		            	
			            	Instance instance = processLine(buffer);			            	

			            	blockingQueue.put(instance);			            	
			            	buffer = br.readLine();
			            	if (buffer == null ) {
			            		this.endOfFile = true;
			            		break;
			            	}
			            }
		            
	            	}		            

		        } catch (Exception e) {
		            e.printStackTrace();
		        }finally{
		            try {
		                br.close();
		            } catch (IOException e) {
		                e.printStackTrace();
		            }
		        }


		  }


		  public int getD() {
			return d;
		}

		public void setD(int d) {
			this.d = d;
		}

		public int getN() {
			return n;
		}

		public void setN(int n) {
			this.n = n;
		}

		public int getM() {
			return m;
		}

		public void setM(int m) {
			this.m = m;
		}

		synchronized public boolean isEndOfFile() {
			return this.endOfFile;			
		}
		
		protected Instance processLine(String line ){
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
					indexy++;
				} else {  // features
					x[indexx] = new AVPair();
					x[indexx].index = Integer.parseInt(tokens[0])-1;         // the indexing starts at 0
					x[indexx++].value = Double.parseDouble(tokens[1]);							
				}
			}
			
			Arrays.sort(y);
			
			return new Instance(x, y);

		  }
		}	

	protected void finalize() {
		this.close();
	}
	
	public void close() {
		if ( ( this.readerthread != null ) && (this.readerthread.endOfFile==false) ){
			this.readerthread.flag = false;
			this.getNextInstance();
		}
	}
	
}
