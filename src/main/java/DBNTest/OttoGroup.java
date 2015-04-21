package DBNTest;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class OttoGroup {
	
	
	
	public static void main(String[] args){
		try {
			getData();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void getData() throws IOException {
		
		File f = new File("train_orig.csv");
		InputStream fis = new FileInputStream(f);
		List<String> lines = IOUtils.readLines(fis);
		INDArray data = Nd4j.ones(lines.size(), 93);
		List<String> outcomeTypes = new ArrayList<>(Arrays.asList("1","2","3","4","5","6","7","8","9"));
		double[][] outcomes = new double[lines.size()][9];
		
		for(int i = 0; i < lines.size(); i++) {
			   String line = lines.get(i);
			  
			   String[] split = line.split(",");
			   
			
			    
			   double[] vector = new double[94];
			   for(int ii = 1; ii < 94; ii++)
			        vector[ii-1] = Double.parseDouble(split[ii]);
//			   System.out.println(vector[0]+" "+vector[1]+" "+vector[2]+" "+vector[3]);
			        data.putRow(i,Nd4j.create(vector));
			
			  String outcome = split[94];
			  if(!outcomeTypes.contains(outcome))
			  { outcomeTypes.add(outcome);}
			
			  double[] rowOutcome = new double[9];
			  rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
//			  System.out.println(outcome+" "+rowOutcome[Integer.valueOf(outcome)-1]);
			  outcomes[i] = rowOutcome;
			  
			  }
			
			 DataSet completedData = new DataSet(data, Nd4j.create(outcomes));

			 completedData.shuffle();
			 SplitTestAndTrain splitData= completedData.splitTestAndTrain( (int) (lines.size()*0.8));
			 
			    RandomGenerator gen = new MersenneTwister(123);
			    NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(5e-1f).constrainGradientToUnitNorm(false).iterations(1000)
                .withActivationType(NeuralNetConfiguration.ActivationType.SAMPLE)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(93).nOut(9).build();
			    
			    
			    
			    
			    DBN dbn = new DBN.Builder().configure(conf).hiddenLayerSizes(new int[]{50,9}).build();
			    	  dbn.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
			    	dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

			 DataSet train = splitData.getTrain().sample(15000);
			
			 dbn.fit(train);
			 System.out.println();
			 BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("otto-dbn.bin"));
			 dbn.write(bos);
			 bos.flush();
			 bos.close();
			 System.out.println("Saved dbn");
			 Evaluation eval = new Evaluation();
			 INDArray output = dbn.output(train.getFeatureMatrix());
			 eval.eval(train.getLabels(),output);
			 System.out.printf("Score: %s\n", eval.stats());
			 System.out.println(("Score " + eval.accuracy()));
			 eval = new Evaluation();
			 output = dbn.output(splitData.getTest().sample(20000).getFeatureMatrix());
			 eval.eval(splitData.getTest().getLabels(),output);
			 System.out.printf("Score: %s\n", eval.stats());
			 System.out.println(("Score " + eval.accuracy()));
			 
			 
	}
}
