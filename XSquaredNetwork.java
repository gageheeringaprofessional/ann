import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

/**@author Gage Heeringa
 * Completed 2/12/2016
 * CS 4811 - Program 1
 * 
 * A feed-forward neural network of an input, a hidden, and an output layer of units.
 * The network uses back-propagation to classify (x,y) coordinates as (1) on or above the y = x^2 parabola,
 * OR (2) below the parabola.
 */
public class XSquaredNetwork extends MultilayerPerceptron{

	final int restartLimit = 4; // number of random restarts the network is allowed
	final int roundLimit = 15000; // numbers of rounds of learning the network is allowed to learn the function

	/**Main
	 */
	public static void main(String args[]){
		XSquaredNetwork x = new XSquaredNetwork(2, 5, 1); // make network and specify the number of units in each layer 
		XSquaredExample[] examples = new XSquaredExample[]{ // examples for learning
							
				// y >= x^2 ?  (Points generated using TrainingSetGenerator)
				//x:[-4,4), 200 tests, 5 units
				new XSquaredExample(2.7,2.2, 0), new XSquaredExample(1.4,-1.8, 0), new XSquaredExample(1.7,0.5, 1), new XSquaredExample(-1.3,1.3, 0), 
				new XSquaredExample(-2.4,0.1, 0), new XSquaredExample(-1.0,1.1, 0), new XSquaredExample(3.1,-2.1, 0), new XSquaredExample(16.4,-4.0, 1), 
				new XSquaredExample(1.9,-2.1, 0), new XSquaredExample(16.3,3.8, 1), new XSquaredExample(18.0,4.0, 1), new XSquaredExample(2.4,-1.4, 1), 
				new XSquaredExample(3.6,-1.4, 1), new XSquaredExample(5.1,-2.8, 0), new XSquaredExample(14.5,3.8, 1), new XSquaredExample(1.4,2.0, 0), 
				new XSquaredExample(8.6,2.9, 1), new XSquaredExample(2.8,1.0, 1), new XSquaredExample(8.4,-3.1, 0), new XSquaredExample(0.1,1.3, 0), 
				new XSquaredExample(0.6,-0.4, 1), new XSquaredExample(15.3,3.7, 1), new XSquaredExample(-1.9,-0.6, 0), new XSquaredExample(3.9,2.3, 0), 
				new XSquaredExample(10.8,-3.2, 1), new XSquaredExample(1.4,-1.1, 1), new XSquaredExample(12.7,3.5, 1), new XSquaredExample(3.4,1.3, 1), 
				new XSquaredExample(1.0,-0.3, 1), new XSquaredExample(-1.5,0.4, 0), new XSquaredExample(4.6,1.7, 1), new XSquaredExample(1.6,-2.0, 0), 
				new XSquaredExample(0.1,-1.6, 0), new XSquaredExample(10.2,3.5, 0), new XSquaredExample(-1.3,0.1, 0), new XSquaredExample(4.4,-2.7, 0), 
				new XSquaredExample(1.2,0.6, 1), new XSquaredExample(12.5,3.5, 1), new XSquaredExample(2.0,0.9, 1), new XSquaredExample(0.8,-0.5, 1), 
				new XSquaredExample(6.7,-2.5, 1), new XSquaredExample(0.4,1.4, 0), new XSquaredExample(1.8,0.8, 1), new XSquaredExample(14.1,-3.7, 1), 
				new XSquaredExample(0.1,1.4, 0), new XSquaredExample(8.2,-2.5, 1), new XSquaredExample(4.1,1.9, 1), new XSquaredExample(0.0,1.7, 0), 
				new XSquaredExample(7.9,-3.2, 0), new XSquaredExample(0.2,1.1, 0), new XSquaredExample(1.1,1.7, 0), new XSquaredExample(14.3,-3.6, 1), 
				new XSquaredExample(-1.6,-0.9, 0), new XSquaredExample(16.8,-3.9, 1), new XSquaredExample(1.0,0.8, 1), new XSquaredExample(-1.0,-1.1, 0), 
				new XSquaredExample(-1.0,-1.1, 0), new XSquaredExample(8.9,2.8, 1), new XSquaredExample(2.8,1.1, 1), new XSquaredExample(1.9,-1.1, 1), 
				new XSquaredExample(3.6,-1.5, 1), new XSquaredExample(5.3,1.9, 1), new XSquaredExample(2.2,-2.2, 0), new XSquaredExample(6.1,2.1, 1), 
				new XSquaredExample(13.9,-3.5, 1), new XSquaredExample(1.4,1.8, 0), new XSquaredExample(2.2,0.8, 1), new XSquaredExample(1.8,0.5, 1), 
				new XSquaredExample(3.1,-1.1, 1), new XSquaredExample(4.5,-2.4, 0), new XSquaredExample(8.5,-3.2, 0), new XSquaredExample(-1.5,0.8, 0), 
				new XSquaredExample(1.0,-0.1, 1), new XSquaredExample(6.2,-2.8, 0), new XSquaredExample(-0.2,-1.5, 0), new XSquaredExample(14.9,3.6, 1), 
				new XSquaredExample(1.0,1.5, 0), new XSquaredExample(13.8,-3.9, 0), new XSquaredExample(7.2,-3.0, 0), new XSquaredExample(-2.0,-0.9, 0), 
				new XSquaredExample(11.7,3.2, 1), new XSquaredExample(13.0,-3.6, 1), new XSquaredExample(5.3,2.7, 0), new XSquaredExample(5.4,-2.2, 1), 
				new XSquaredExample(2.1,-1.9, 0), new XSquaredExample(9.5,-3.5, 0), new XSquaredExample(3.7,-1.7, 1), new XSquaredExample(12.9,3.9, 0), 
				new XSquaredExample(2.1,2.2, 0), new XSquaredExample(12.1,3.4, 1), new XSquaredExample(1.3,-0.1, 1), new XSquaredExample(0.5,-0.1, 1), 
				new XSquaredExample(-1.3,-0.6, 0), new XSquaredExample(15.7,-3.8, 1), new XSquaredExample(15.6,-3.8, 1), new XSquaredExample(6.1,-2.2, 1), 
				new XSquaredExample(2.3,1.3, 1), new XSquaredExample(-1.6,0.1, 0), new XSquaredExample(10.2,2.9, 1), new XSquaredExample(12.9,-3.4, 1), 
				new XSquaredExample(10.9,-3.0, 1), new XSquaredExample(2.3,0.8, 1), new XSquaredExample(6.9,-2.6, 1), new XSquaredExample(14.6,-3.7, 1), 
				new XSquaredExample(-1.7,-1.0, 0), new XSquaredExample(1.6,-1.1, 1), new XSquaredExample(5.2,-1.9, 1), new XSquaredExample(0.1,-0.3, 1), 
				new XSquaredExample(2.7,1.3, 1), new XSquaredExample(-1.5,-0.9, 0), new XSquaredExample(2.1,-1.8, 0), new XSquaredExample(2.8,-2.2, 0), 
				new XSquaredExample(-1.6,0.9, 0), new XSquaredExample(2.2,2.2, 0), new XSquaredExample(1.5,-0.6, 1), new XSquaredExample(1.1,-2.0, 0), 
				new XSquaredExample(0.8,0.3, 1), new XSquaredExample(4.1,-2.6, 0), new XSquaredExample(2.8,2.2, 0), new XSquaredExample(-1.5,0.4, 0), 
				new XSquaredExample(11.1,3.3, 1), new XSquaredExample(-2.4,0.5, 0), new XSquaredExample(1.7,0.6, 1), new XSquaredExample(16.0,-3.9, 1), 
				new XSquaredExample(10.3,-3.5, 0), new XSquaredExample(16.3,3.8, 1), new XSquaredExample(-0.3,-0.9, 0), new XSquaredExample(8.0,2.5, 1), 
				new XSquaredExample(4.1,2.5, 0), new XSquaredExample(12.4,-3.5, 1), new XSquaredExample(13.2,-4.0, 0), new XSquaredExample(-0.8,-1.0, 0), 
				new XSquaredExample(-2.9,-0.0, 0), new XSquaredExample(-0.4,1.4, 0), new XSquaredExample(-0.9,0.4, 0), new XSquaredExample(5.3,2.7, 0), 
				new XSquaredExample(2.9,2.1, 0), new XSquaredExample(10.8,-3.7, 0), new XSquaredExample(13.4,3.9, 0), new XSquaredExample(4.8,-2.7, 0), 
				new XSquaredExample(1.2,1.7, 0), new XSquaredExample(-2.6,-0.4, 0), new XSquaredExample(0.2,1.6, 0), new XSquaredExample(-2.6,-0.2, 0), 
				new XSquaredExample(5.3,2.0, 1), new XSquaredExample(5.4,-2.1, 1), new XSquaredExample(1.1,-0.6, 1), new XSquaredExample(5.5,2.1, 1), 
				new XSquaredExample(15.5,3.7, 1), new XSquaredExample(11.7,3.6, 0), new XSquaredExample(6.2,-2.7, 0), new XSquaredExample(-1.0,-1.4, 0), 
				new XSquaredExample(6.5,2.4, 1), new XSquaredExample(-1.2,-0.2, 0), new XSquaredExample(7.6,-3.1, 0), new XSquaredExample(4.2,-1.6, 1), 
				new XSquaredExample(11.3,-3.1, 1), new XSquaredExample(10.3,2.9, 1), new XSquaredExample(4.1,2.5, 0), new XSquaredExample(-0.2,1.1, 0), 
				new XSquaredExample(-2.2,0.2, 0), new XSquaredExample(5.4,2.2, 1), new XSquaredExample(4.3,2.4, 0), new XSquaredExample(11.0,-3.3, 1), 
				new XSquaredExample(5.5,2.2, 1), new XSquaredExample(6.9,3.0, 0), new XSquaredExample(0.3,-1.3, 0), new XSquaredExample(9.2,2.9, 1), 
				new XSquaredExample(7.7,3.0, 0), new XSquaredExample(1.5,-0.4, 1), new XSquaredExample(8.0,-3.2, 0), new XSquaredExample(-0.5,-1.0, 0), 
				new XSquaredExample(-1.6,-0.8, 0), new XSquaredExample(3.2,-2.1, 0), new XSquaredExample(6.0,2.1, 1), new XSquaredExample(3.5,1.7, 1), 
				new XSquaredExample(-1.7,-0.1, 0), new XSquaredExample(14.3,-3.7, 1), new XSquaredExample(12.3,-3.4, 1), new XSquaredExample(1.9,0.6, 1), 
				new XSquaredExample(1.1,-1.5, 0), new XSquaredExample(6.4,-2.5, 1), new XSquaredExample(3.3,1.3, 1), new XSquaredExample(5.2,-2.8, 0), 
				new XSquaredExample(1.5,2.0, 0), new XSquaredExample(2.2,-1.1, 1), new XSquaredExample(12.8,-3.9, 0), new XSquaredExample(5.4,1.9, 1), 
				new XSquaredExample(11.9,-3.8, 0), new XSquaredExample(0.6,-0.7, 1), new XSquaredExample(1.8,0.2, 1), new XSquaredExample(4.5,2.4, 0), 
				new XSquaredExample(2.4,1.2, 1), new XSquaredExample(15.5,3.9, 1), new XSquaredExample(3.6,-1.7, 1), new XSquaredExample(0.0,-1.3, 0), 
				new XSquaredExample(15.4,-3.8, 1), new XSquaredExample(16.1,-3.9, 1), new XSquaredExample(3.1,1.6, 1), new XSquaredExample(1.7,0.1, 1)
				
				
		};

		x.randomRestartTesting(examples); // learning and checking
		//x.generatePoints(); 
		//x.generatePoints(examples); //only graph the training set
	}

	/* Constructor */
	XSquaredNetwork(int inputSize, int hiddenSize, int outputSize){
		super(inputSize, hiddenSize, outputSize);
	}

	/**Test the network.
	 * Do a random restart if the network does not successfully learn XOR by the specified number of rounds.
	 */
	void randomRestartTesting(XSquaredExample[] examples){
		int success = -1; // see if the network learns all the examples
		for(int i = 0; i <= restartLimit; i++){ // give it some number of chances
			if(i > 0){
				System.out.printf("---------------- (RANDOM RESTART #%d) ----------------\n", i);
			}
			success = test(examples); 
			// if the network reaches the round limit for learning, do random restart
			if(success != 0){
				randomRestart();
			}
			else{
				return;
			}
		}

		if(success != 0){
			System.out.printf("The network could not learn with %d rounds and %d random restarts.\n",
					roundLimit, restartLimit);
		}
	}

	/**Test the network.
	 * Return 0 if the network successfully learned, else -1.
	 * Print details on the network's final configurations.
	 */
	int test(XSquaredExample[] examples){

		/* train the network by giving examples of correct classifications */
		for(int i = 0; i < roundLimit; i++){
			double a = examples[i % examples.length].a; //cycling through the tests
			double  b = examples[i % examples.length].b;
			int answer = examples[i % examples.length].c;
			double[] x = new double[]{a, b}; //input vector
			int[] y = new int[]{answer}; //correct hypothesis vector
			training(x, y);

			/* learning check: confirm the network produces correct output for all examples */
			boolean done = true;
			for(int j = 0; j < examples.length; j++){
				a = examples[j].a;
				b = examples[j].b;
				answer = examples[j].c;
				x = new double[]{a, b}; //input vector

				int results[] = applyKnowledge(x); //hard threshold function is used to classify answer as 0 or 1
				if(results[0] != answer){
					done = false;
					break;
				}
			}
			if(done){
				//print information about the successful network
				System.out.printf("Network completed learning after %d rounds. (%d tests and %d hidden units)\n",
						i + 1, examples.length, hiddenSize);
				System.out.println("\tWeights between input and hidden units:");
				for(int k = 0; k < inputSize; k++){ 
					for(int j = inputSize; j < inputSize + hiddenSize; j++){
						System.out.printf("W[%d][%d]=%.4f\n", k, j, W[k][j]);
					}
				}
				System.out.println("\tWeights between hidden and output units:");
				for(int k = inputSize; k < inputSize + hiddenSize; k++){ 
					for(int j = inputSize + hiddenSize; j < nrUnits; j++){
						System.out.printf("W[%d][%d]=%.4f\n", k, j, W[k][j]);
					}
				}
				break;
			}
			if(i == roundLimit - 1){
				int c0 = 0; //those with answer 0/1 that were classified right
				int  c1 = 0;

				System.out.printf("Network HALTED LEARNING after %d rounds.\n", i + 1);
				for(int j = 0; j < examples.length; j++){
					a = examples[j].a;
					b = examples[j].b;
					answer = examples[j].c;
					x = new double[]{a, b}; //input vector

					int results[] = applyKnowledge(x); //hard threshold function is used to classify answer as 0 or 1

					//classified wrong
					//if(results[0] != answer){
					//System.out.printf("\tIs %.1f >= %.1f^2?  Result: <%d>  Answer: <%d>\n", a, b, results[0], answer);
					//}

					if(results[0] == answer && answer == 1){
						c1++;
					}
					if(results[0] == answer && answer == 0){
						c0++;
					}
				}
				//print information about the failed network
				System.out.println("\tWeights between input and hidden units:");
				for(int k = 0; k < inputSize; k++){ 
					for(int j = inputSize; j < inputSize + hiddenSize; j++){
						System.out.printf("W[%d][%d]=%.4f  ", k, j, W[k][j]);
					}
					System.out.println();
				}
				System.out.println("\tWeights between hidden and output units:");
				for(int k = inputSize; k < inputSize + hiddenSize; k++){ 
					for(int j = inputSize + hiddenSize; j < nrUnits; j++){
						System.out.printf("W[%d][%d]=%.4f  ", k, j, W[k][j]);
					}
					System.out.println();
				}

				/* accept some percentage of accurate classification */
				double perc = (double)(c1 + c0) / (double)examples.length;
				System.out.printf("*** %.3f accuracy (%d/%d examples classified correctly) ***\n", 
						perc, c0 + c1, examples.length); 
				if(perc >= .88){ 
					System.out.printf("*** %.3f >= %.2f classification accuracy. Accepting network state. ***\n",
							perc, .88);
					return 0;
				}
				return -1;
			}
		}

		return 0;
	}

	/**This generates (x,y) coordinates for graphing data points to depict the function that
	 * the network has learned.
	 * 
	 * NOTE: The network is currently configured to do learning with x values in range [-4,4],
	 * therefore points generated are for x in range [-4,4) with .1 increments and
	 * y in range [-2, 4^2=16] with .1 increments.  It's tested rather the (x,y) points are 
	 * classified above or on the parabola (y >= x^2), OR below it (y < x^2).
	 * So this is 14,480 points.
	 * 
	 * Two files are generated in the same directory as where the program is run:
	 * "generatedGEQPoints.txt" contains coordinates (x,y) such that y >= x^2.
	 * "generatedLPoints.txt" contains coordinates (x,y) such that y < x^2.
	 */
	void generatePoints(){
		ArrayList<String> geqPoints = new ArrayList<String>();
		ArrayList<String> lPoints = new ArrayList<String>();
		for(double i = -4.0; i <= 4.0; i += 0.1){
			for(double j = -2.0; j <= 16.0; j += 0.1){
				int results[] = applyKnowledge(new double[]{j, i}); //y, x format when computed...
				if(results[0] == 1){
					geqPoints.add(String.format("%.1f, %.1f", i, j)); //(x,y) coordinate
				}
				else{
					lPoints.add(String.format("%.1f, %.1f", i, j)); //(x,y) coordinate
				}
			}
		}

		// y >= x^2
		PrintWriter out = null;
		try{
			out = new PrintWriter(new File("generatedGEQPoints.txt"));
		} catch(FileNotFoundException e){
			e.printStackTrace();
		}
		for(int i = 0; i < geqPoints.size(); i++){
			out.write(geqPoints.get(i) + "\n");
		}
		out.close();

		// y < x^2
		try{
			out = new PrintWriter(new File("generatedLPoints.txt"));
		} catch(FileNotFoundException e){
			e.printStackTrace();
		}
		for(int i = 0; i < lPoints.size(); i++){
			out.write(lPoints.get(i) + "\n");
		}
		out.close();
		
		System.out.println(geqPoints.size() + lPoints.size() + " points");
	}
	
	/**This generates (x,y) coordinates for graphing the training set to depict the function that
	 * the network has learned.
	 * 
	 * Two files are generated in the same directory as where the program is run:
	 * "generatedGEQPoints.txt" contains coordinates (x,y) such that y >= x^2.
	 * "generatedLPoints.txt" contains coordinates (x,y) such that y < x^2.
	 */
	void generatePoints(XSquaredExample[] examples){
		ArrayList<String> geqPoints = new ArrayList<String>();
		ArrayList<String> lPoints = new ArrayList<String>();
		for(int i = 0; i < examples.length; i ++){
			int results[] = applyKnowledge(new double[]{examples[i].a, examples[i].b}); //y, x format when computed...
			if(results[0] == 1){
				geqPoints.add(String.format("%f, %f", examples[i].b, examples[i].a)); //(x,y) coordinate
			}
			else{
				lPoints.add(String.format("%f, %f", examples[i].b, examples[i].a)); //(x,y) coordinate
			}
	}

		// y >= x^2
		PrintWriter out = null;
		try{
			out = new PrintWriter(new File("generatedGEQPoints.txt"));
		} catch(FileNotFoundException e){
			e.printStackTrace();
		}
		for(int i = 0; i < geqPoints.size(); i++){
			out.write(geqPoints.get(i) + "\n");
		}
		out.close();

		// y < x^2
		try{
			out = new PrintWriter(new File("generatedLPoints.txt"));
		} catch(FileNotFoundException e){
			e.printStackTrace();
		}
		for(int i = 0; i < lPoints.size(); i++){
			out.write(lPoints.get(i) + "\n");
		}
		out.close();
		
		System.out.println(geqPoints.size() + lPoints.size() + " points");
	}
}

/**Example for training the network. (y,x) = inputs, passed = 1 IF y >= x^2, ELSE 0
 */
class XSquaredExample{
	double a, b;
	int c;
	XSquaredExample(double y, double x, int passed){
		a = y;
		b = x;
		c = passed;
	}
}