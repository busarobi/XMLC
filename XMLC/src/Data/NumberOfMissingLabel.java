package Data;

import java.util.TreeSet;

import IO.DataReader;

public class NumberOfMissingLabel {

	public NumberOfMissingLabel() {
		// TODO Auto-generated constructor stub
	}

	public static void main(String[] args) throws Exception {
		String[] dirs = {
				"/Users/busarobi/work/XMLC/data/RCV1/",
				//"/home/busarobi/XMLC/data/RCV1/",
				"/home/busarobi/XMLC/data/wiki10/",
				"/home/busarobi/XMLC/data/WikiLSHTC/",
				"/home/busarobi/XMLC/data/AmazonCat/",
				"/home/busarobi/XMLC/data/Amazon/",
				"/home/busarobi/XMLC/data/Delicious-200K/",
		};

		String[] trainFiles = { 
				"rcv1x_train.txt",
				"wiki10_train.txt",
				"wikiLSHTC_train.txt",
				"amazonCat_train.txt",
				"amazon_train.txt",
				"deliciousLarge_train.txt"
		};
		
		String[] testFiles = { 
				"rcv1x_test.txt",
				"wiki10_test.txt",
				"wikiLSHTC_test.txt",
				"amazonCat_test.txt",
				"amazon_test.txt",
				"deliciousLarge_test.txt"
		};
	
		
		for( int i=0; i < trainFiles.length; i++ ){
			String trainFile = dirs[i] + trainFiles[i];

			DataReader datareadertrain = new DataReader(trainFile, false, true);
			AVTable traindata = datareadertrain.read();
			int[] labelDistTrain = AVTable.getNumOfLabels(traindata);
			
			String testFile = dirs[i] + testFiles[i];
			
			DataReader datareadertest = new DataReader(testFile, false, true);
			AVTable testdata = datareadertest.read();
			int[] labelDistTest = AVTable.getNumOfLabels(testdata);
			
			int missingTrain = 0;
			int missingTest = 0;

			int onlyoneTrain = 0;
			int onlyoneTest = 0;

			
			int missingBoth = 0;
			TreeSet<Integer> presentedInTrainAndNotInTest = new TreeSet<Integer>();
			
			for( int j=0; j < labelDistTrain.length; j++ ){
				if (labelDistTrain[j] == 0 ){
					missingTrain++;
				}
				
				if (labelDistTest[j] == 0 ){
					missingTest++;
				}

				if ( (labelDistTrain[j] == 1 ) && (labelDistTest[j] == 0 ) ){
					onlyoneTrain++;
				}
				
				if ( (labelDistTest[j] == 1 ) && (labelDistTrain[j] == 0 ) ){
					onlyoneTest++;
				}
				
				
				
				if ( (labelDistTrain[j] == 0 ) && (labelDistTest[j] == 0 ) ) {
					missingBoth++;
				}	
				
				if ( (labelDistTrain[j]>1) && (labelDistTest[j] == 0 ) ) {
					presentedInTrainAndNotInTest.add(j);
				}
				
			}
			
			System.out.println(trainFile + " " + missingTrain + "(" + labelDistTrain.length + ")" );
			System.out.println(testFile + " " + missingTest + "(" + labelDistTest.length + ")" );

			System.out.println(trainFile + " there is only one: " + onlyoneTrain + "(" + missingTest + ")" );
			System.out.println(testFile + " there is only one: " + onlyoneTest + "(" + missingTrain + ")" );
			
			
			System.out.println( "Missing from both: " + missingBoth + "(" + labelDistTest.length + ")" );
			
			for( Integer j : presentedInTrainAndNotInTest ) {
				System.out.println( "Label: " + j + " Number of occurence in train: " + labelDistTrain[j] );
			}
			System.out.println();
		}
		
	}

}
