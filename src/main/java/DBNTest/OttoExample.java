package DBNTest;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import javax.imageio.stream.FileImageInputStream;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/12/14.
 */
public class OttoExample {


    //private static Logger log = LoggerFactory.getLogger(OttoExample.class);

    public static void main(String[] args) throws FileNotFoundException {
        RandomGenerator gen = new MersenneTwister(123);

        List<NeuralNetConfiguration> conf = new NeuralNetConfiguration.Builder()
        .iterations(200)
        .weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen,1e-3)).constrainGradientToUnitNorm(false)
        .l2(0.01f)
        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).activationFunction(Activations.tanh())
        .rng(gen).regularization(true).visibleUnit(RBM.VisibleUnit.SOFTMAX).hiddenUnit(RBM.HiddenUnit.SOFTMAX)
        .learningRate(1e-3f).momentum(0.9f).nIn(93).nOut(9)
        .list(2).override(new NeuralNetConfiguration.ConfOverride() {
            @Override
            public void override(int i, NeuralNetConfiguration.Builder builder) {
            if (i == 1) {
                builder.weightInit(WeightInit.ZERO);
                builder.activationFunction(Activations.sigmoid());
                builder.lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY);
              }
            }
        })    
        .build();



        DBN d = new DBN.Builder()
                .layerWiseConfiguration(conf)
                .hiddenLayerSizes(new int[]{500})
                .build();


        NeuralNetConfiguration.setClassifier(d.getOutputLayer().conf());
  

        DataSetIterator iter = new OttoDataSetIterator(10, 5000);
//        System.out.println("num examples:"+iter.numExamples()+" "+iter.batch());
//        System.out.println(iter.next());
        int samples = 3000;
        //fetch first
//        DBN d= SerializationUtils.readObject(new File("otto-dbn.bin"));
        DataSet next = iter.next(samples);
        
        
       
        d.pretrain(next.getFeatureMatrix(),2, 1e-3f, 50);
        iter.reset();next = iter.next(samples);
              
       
        SplitTestAndTrain data= next.splitTestAndTrain((int) (samples*0.4));
        d.fit(data.getTrain());
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("otto-dbn.bin"));
		 d.write(bos);
		 try {
			bos.flush();
			bos.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 
		 System.out.println("Saved dbn");

		 
		

		 Evaluation eval = new Evaluation();
        INDArray output = d.output(data.getTrain().getFeatureMatrix());
        eval.eval(data.getTrain().getLabels(),output);
        System.out.println("Score train " + eval.stats());
        System.out.println("Score train " + eval.f1());
        
        DataSet test = data.getTest();
        eval = new Evaluation();
        output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        System.out.println("Score " + eval.f1());
        System.out.println(eval.precision());
        System.out.println(eval.recall());
        
    }
}