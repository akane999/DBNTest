package DBNTest;
import java.io.IOException;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
public class OttoDataFetcher extends BaseDataFetcher {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4834394700169578300L;

	public final static int NUM_EXAMPLES = 61868;
	public OttoDataFetcher() {
		numOutcomes = 9;
		inputColumns = 93;
		totalExamples = NUM_EXAMPLES;
	}
	@Override
	public void fetch(int numExamples) {
		int from = cursor;
		int to = cursor + numExamples;
		if(to > totalExamples)
			to = totalExamples;
		
		try {
			initializeCurrFromList(OttoUtils.loadOtto(from, to));
			cursor += numExamples;
		} catch (IOException e) {
			throw new IllegalStateException("Unable to load otto");
		}
		
	}


}











	
	
	
	

	