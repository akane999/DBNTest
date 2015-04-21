package DBNTest;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

public class OttoDataSetIterator extends BaseDatasetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7411759713419965384L;

	public OttoDataSetIterator(int batch, int numExamples) {
		super(batch, numExamples, new OttoDataFetcher());
	}

}