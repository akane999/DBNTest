package DBNTest;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class OttoUtils {
	final static int feature_size = 93;

	public static List<DataSet> loadOtto(int from, int to) throws IOException {
		File f = new File("train_orig.csv");
		InputStream fis = new FileInputStream(f);
		List<DataSet> list = new ArrayList<>();
		@SuppressWarnings("unchecked")
		List<String> lines = IOUtils.readLines(fis);
		INDArray data = Nd4j.ones(lines.size(), feature_size);
		List<String> outcomeTypes = new ArrayList<>(Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9"));
		double[][] outcomes = new double[lines.size()][9];

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);

			String[] split = line.split(",");

			String[] vector = new String[94];
			for (int ii = 1; ii < 94; ii++) {
				vector[ii - 1] = split[ii];
//				System.out.println(vector[0]+" "+vector[1]+" "+vector[2]+" "+vector[3]);
			}
				OttoUtils.addRow(data, i, vector);

			String outcome = split[94];
			if (!outcomeTypes.contains(outcome)) {
				outcomeTypes.add(outcome);
			}

			double[] rowOutcome = new double[9];
			rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
			// System.out.println(outcome+" "+rowOutcome[Integer.valueOf(outcome)-1]);
			outcomes[i] = rowOutcome;

		}

		DataSet completedData = new DataSet(data, Nd4j.create(outcomes));
		
		completedData.shuffle();
		

		list=completedData.asList();
//		System.out.println(list.get(0).getFeatures().toString());
//		System.out.println(list.get(0).getLabels().toString());
//		System.out.println(list.get(1).getFeatures().toString());
//    	System.out.println(list.get(1).getLabels().toString());
//		System.out.println(list.get(2).getFeatures().toString());
//		System.out.println(list.get(2).getLabels().toString());
//		System.out.println(list.get(3).getFeatures().toString());
//		System.out.println(list.get(3).getLabels().toString());
//		System.out.println(list.get(4).getFeatures().toString());
//		System.out.println(list.get(4).getLabels().toString());
		

		return list.subList(from, to);

	}

	private static void addRow(INDArray ret, int row, String[] line) {
		double[] vector = new double[feature_size];
		for (int i = 0; i < feature_size; i++)
			vector[i] = Double.parseDouble(line[i]);

		ret.putRow(row, Nd4j.create(vector));
	}
}
