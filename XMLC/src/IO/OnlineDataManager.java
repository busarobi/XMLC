package IO;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import Data.AVPair;
import Data.Instance;

public class OnlineDataManager extends DataManager {
	protected ReaderThread  readerthread = null;
	protected String filename = null;
	protected BlockingQueue<Instance> blockingQueue = null;
	protected int bufferSize = 16384;
	protected Thread rthread = null;
	protected int processedItem = 0;
	public OnlineDataManager(String filename) {
		this.filename = filename;		
		this.blockingQueue = new ArrayBlockingQueue<Instance>(this.bufferSize);
		this.readerthread = new ReaderThread(this.blockingQueue, this.filename);
		this.rthread = new Thread(this.readerthread);
		this.rthread.start();		
	}

	@Override
	public boolean hasNext() {				
		return ((this.blockingQueue.size() > 0) || (! this.readerthread.isEndOfFile() ));
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
		return this.readerthread.getD();
	}

	@Override
	public int getNumberOfLabels() {		
		return this.readerthread.getM();
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
		  protected String filename = null;
		  protected int d; 
		  protected int n;
		  protected int m;
		  protected int ni = 0;
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
				this.ni = 0;
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
		            String buffer =null;
		            while((buffer=br.readLine())!=null){
		            	if (this.flag == false ) break;
		            	ni++;
		            	Instance instance = processLine(buffer);
		                blockingQueue.put(instance);		                
		            }
		            //blockingQueue.put(null);  //When end of file has been reached
		            this.endOfFile = true;

		        } catch (FileNotFoundException e) {

		            e.printStackTrace();
		        } catch (IOException e) {

		            e.printStackTrace();
		        } catch(InterruptedException e){

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
			//return this.endOfFile;
			return (this.ni >= this.n);
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
