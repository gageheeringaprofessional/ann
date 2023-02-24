import java.io.OutputStream;
import java.io.PrintStream;

/**@author Gage Heeringa
 * 2/6/2016
 * CS 4811 - Program 1
 * 
 * A structure for a feed-forward neural network that consists of an input, a hidden, and an output layer of units.
 * Back-propagation is used for learning.
 * 
 * Only difference with this MLP and the one turned in with XorNetwork is that this MLP class allows for training
 * with input vector "x" containing doubles (rather than just integers).
 */
public class MultilayerPerceptron{

	/* Globals */
	PrintStream ps; //System.out

	final int BIAS = 1; //"magnitude of the threshold"
	final int ALPHA = 1; //= learning rate = step size
	int inputSize; //# units in input layer
	int hiddenSize;
	int outputSize;
	int nrUnits; //# units in network
	double a[]; //activation values a.k.a. unit values
	double W[][]; //weights (lower index points to higher, so W[lower][higher] will have the weight
	//W[i][i] = unit bias
	double D[]; //delta values for each unit. they're used for updating the weights

	/* Constructor */
	MultilayerPerceptron(int inputSize, int hiddenSize, int outputSize){
		ps = System.out;

		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		nrUnits = inputSize + hiddenSize + outputSize;
		a = new double[nrUnits]; 
		W = new double[nrUnits][nrUnits];
		D = new double[nrUnits];
		//assign initial random weights
		for(int i = 0; i < W.length; i++){
			for(int j = 0; j < W[i].length; j++){
				if( (int)(Math.random()*2)/*[0,1]*/ == 1){
					W[i][j] = (int)(Math.random()*2); /*[0,2]*/
				}
				else{
					W[i][j] = -1*(int)(Math.random()*2); /*[-2,0]*/
				}
			}
		}
		/*
		//Dr. Onder's NXOR example:
		W[0][2] = 2; //W-I1-H1
		W[0][3] = 1; //W-I1-H2
		W[1][2] = -2; //W-I2-H1
		W[1][3] = 3; //W-I2-H2
		W[2][4] = 3; //W-H1-T1
		W[3][4] = -2; //W-H2-T1
		W[0][0] = 1;
		W[1][1] = 1;
		W[2][2] = 0; //H1
		W[3][3] = -1;
		W[4][4] = -1; //T1
		 */
	}

	/**Given an input vector "x" and correct hypothesis vector "y", perform one round of back-propagation learning.
	 */
	void training(int[] x, int[] y){
		forwardpropagate(x, y); //major step 1
		backpropagate(); //major step 2
	}
	
	/**Given an input vector "x" and correct hypothesis vector "y", perform one round of back-propagation learning.
	 */
	void training(double[] x, int[] y){
		forwardpropagate(x, y); //major step 1
		backpropagate(); //major step 2
	}

	/**Given input vector "x", propagate x forward to compute a_j = g(in_j) for each unit.
	 * Given correct hypothesis vector "y", compute the error for the output layer to update its delta values.
	 * 
	 * Return the output vector.  
	 * (a_j = the unit j's value) (g = the activation function) (in = the unit input function)
	 */
	double[] forwardpropagate(int x[], int y[]){
		disablePrinting();
		/* input layer ( a_i = x_i ) */
		for(int i = 0; i < inputSize; i++){
			a[i] = x[i];
		}

		/* hidden layer ( a_j = g(in_j) ) */
		for(int j = inputSize; j < inputSize + hiddenSize; j++){
			System.out.println("I-H j=" + j);
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_i) over i
			= sum of weights*activations (between unit j and units in previous layer) */
			for(int i = 0; i < inputSize; i++){
				in_j += W[i][j]*a[i];
			}
			System.out.println("   in_j=" + in_j);
			a[j] = g(in_j); //apply activation function "g" to unit input function for j
			System.out.println("   sigmoid=" + a[j]);
		}

		/* output layer ( a_j = g(in_j) ) */
		for(int j = inputSize + hiddenSize; j < nrUnits; j++){
			System.out.println("H-T j=" + j);
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_i) over i
			= sum of weights*activations (between unit j and units in previous layer) */
			for(int i = inputSize; i < inputSize + hiddenSize; i++){
				in_j += W[i][j]*a[i];
			}
			System.out.println("  in_j=" + in_j);
			a[j] = g(in_j); //apply activation function "g" to unit input function for j
			System.out.println("  sigmoid=" + a[j]);
		}

		/* sanity check: for input layer, a_i = x_i
						 for hidden/output layers, a_j = g(in_j), and they are now proper hypothesis vectors */

		/* output vector */
		double ret[] = new double[outputSize];
		for(int i = nrUnits - outputSize; i < nrUnits; i++){
			ret[i - (nrUnits - outputSize)] = a[i];
		}

		/* compute error and deltas at output layer */
		double[] err = new double[nrUnits]; //y_j - a_j = correct vector - hypothesis vector = error vector for the output layer
		for(int k = inputSize + hiddenSize; k < nrUnits; k++){
			err[k] = y[k - (inputSize + hiddenSize)] - a[k]; 
			//NOTE: our activation function has the property that g'(x) = g(x)*( 1 - g(x) )
			D[k] = a[k]*(1- a[k])*err[k]; //g'(in_k) * (correct - hypothesis) = modified error = delta_k 
			System.out.printf("D[%d] = %f\n", k, D[k]);
		}

		enablePrinting();
		return ret;
	}
	
	/**Given input vector "x", propagate x forward to compute a_j = g(in_j) for each unit.
	 * Given correct hypothesis vector "y", compute the error for the output layer to update its delta values.
	 * 
	 * Return the output vector.  
	 * (a_j = the unit j's value) (g = the activation function) (in = the unit input function)
	 */
	double[] forwardpropagate(double x[], int y[]){
		disablePrinting();
		/* input layer ( a_i = x_i ) */
		for(int i = 0; i < inputSize; i++){
			a[i] = x[i];
		}

		/* hidden layer ( a_j = g(in_j) ) */
		for(int j = inputSize; j < inputSize + hiddenSize; j++){
			System.out.println("I-H j=" + j);
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_i) over i
			= sum of weights*activations (between unit j and units in previous layer) */
			for(int i = 0; i < inputSize; i++){
				in_j += W[i][j]*a[i];
			}
			System.out.println("   in_j=" + in_j);
			a[j] = g(in_j); //apply activation function "g" to unit input function for j
			System.out.println("   sigmoid=" + a[j]);
		}

		/* output layer ( a_j = g(in_j) ) */
		for(int j = inputSize + hiddenSize; j < nrUnits; j++){
			System.out.println("H-T j=" + j);
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_i) over i
			= sum of weights*activations (between unit j and units in previous layer) */
			for(int i = inputSize; i < inputSize + hiddenSize; i++){
				in_j += W[i][j]*a[i];
			}
			System.out.println("  in_j=" + in_j);
			a[j] = g(in_j); //apply activation function "g" to unit input function for j
			System.out.println("  sigmoid=" + a[j]);
		}

		/* sanity check: for input layer, a_i = x_i
						 for hidden/output layers, a_j = g(in_j), and they are now proper hypothesis vectors */

		/* output vector */
		double ret[] = new double[outputSize];
		for(int i = nrUnits - outputSize; i < nrUnits; i++){
			ret[i - (nrUnits - outputSize)] = a[i];
		}

		/* compute error and deltas at output layer */
		double[] err = new double[nrUnits]; //y_j - a_j = correct vector - hypothesis vector = error vector for the output layer
		for(int k = inputSize + hiddenSize; k < nrUnits; k++){
			err[k] = y[k - (inputSize + hiddenSize)] - a[k]; 
			//NOTE: our activation function has the property that g'(x) = g(x)*( 1 - g(x) )
			D[k] = a[k]*(1- a[k])*err[k]; //g'(in_k) * (correct - hypothesis) = modified error = delta_k 
			System.out.printf("D[%d] = %f\n", k, D[k]);
		}

		enablePrinting();
		return ret;
	}

	/**Back-propagate the delta values from output layer -> hidden layer -> input layer.
	 */
	void backpropagate(){
		disablePrinting();

		//"for l = L - 1 to 1 do"
		//hidden layer
		for(int i = inputSize; i < inputSize + hiddenSize; i++){
			// at this point, a[j] = g(in_j)
			double sum = 0; //Sum( W_i,j * Delta_j ) over j, so concerns units i is pointing to
			for(int j = inputSize + hiddenSize; j < nrUnits; j++){
				sum += W[i][j]*D[j];
			}
			//recall: g'(x) = g(x)*( 1 - g(x) )
			D[i] = a[i]*(1- a[i])*sum; //D_i <- g'(in_i)*Sum( W_i,j * Delta_j )
		}

		//input layer
		for(int i = 0; i < inputSize; i++){
			// at this point, a[j] = g(in_j)
			double sum = 0; //Sum( W_i,j * Delta_j ) over j, so concerns units i is pointing to
			for(int j = inputSize; j < inputSize + hiddenSize; j++){
				sum += W[i][j]*D[j];
			}
			//recall: g'(x) = g(x)*( 1 - g(x) )
			D[i] = a[i]*(1- a[i])*sum; //D_i <- g'(in_i)*Sum( W_i,j * Delta_j )
		}

		for(int i = 0; i < D.length; i++){
			System.out.printf("D[%d] = %f\n", i, D[i]);
		}

		/* adjust all the weights */
		for(int i = 0; i < W.length; i++){
			for(int j = 0; j < W[i].length; j++){
				W[i][j] += ALPHA * a[i] * D[j];
			}
		}

		//print weights - also prints irrelevant weights (between input/output, units in same layer, etc.)
		for(int i = 0; i < W.length; i++){ 
			for(int j = i; j < W.length; j++){
				if(i != j && W[i][j] != 0){
					System.out.printf("W[%d][%d]=%.4f\n", i, j, W[i][j]);
				}
			}
		}

		//print bias
		for(int i = 0; i < W.length; i++){ 
			System.out.printf("bias[%d][%d]=%.4f\n", i, i, W[i][i]);
		}

		enablePrinting();
	}

	/**Once the network has learned, use it with this function.
	 */
	int[] applyKnowledge(int x[]){
		/* input layer ( a_i = x_i ) */
		for(int i = 0; i < inputSize; i++){
			a[i] = x[i];
		}

		/* hidden layer ( a_j = g(in_j) ) */
		for(int j = inputSize; j < inputSize + hiddenSize; j++){
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_j) over i
			= sum of weights*activations (between unit j and units in previous layer) = vector w . vector x */
			for(int i = 0; i < inputSize; i++){
				in_j += W[i][j]*a[i];
			}

			/* apply hard threshold function to unit input function for j */
			a[j] = g(in_j);
		}

		/* output layer ( a_j = Threshold(in_j) ) */
		int ret[] = new int[outputSize]; //output vector
		int reti = 0;
		for(int j = inputSize + hiddenSize; j < nrUnits; j++){
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_j) over i
			= sum of weights*activations (between unit j and units in previous layer) = vector w . vector x */
			for(int i = inputSize; i < inputSize + hiddenSize; i++){
				in_j += W[i][j]*a[i];
			}

			/* apply hard threshold function to unit input function for j (the hypothesis vector, 
			 * given input vector "x", is 1 if the weight vector dot the input vector is >= 0, else 0 */
			if(in_j >= 0){
				ret[reti++] = 1;
			}
			else{
				ret[reti++] = 0;
			}
		}

		return ret;
	}

	/**Once the network has learned, this function is useful for generating data points
	 * to determine the shape of the network's classification boundary.
	 */
	int[] applyKnowledge(double x[]){
		/* input layer ( a_i = x_i ) */
		for(int i = 0; i < inputSize; i++){
			a[i] = x[i];
		}

		/* hidden layer ( a_j = g(in_j) ) */
		for(int j = inputSize; j < inputSize + hiddenSize; j++){
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_j) over i
			= sum of weights*activations (between unit j and units in previous layer) = vector w . vector x */
			for(int i = 0; i < inputSize; i++){
				in_j += W[i][j]*a[i];
			}

			a[j] = g(in_j); //apply activation function "g" to unit input function for j
		}

		/* output layer ( a_j = Threshold(in_j) ) */
		int ret[] = new int[outputSize]; //output vector
		int reti = 0;
		for(int j = inputSize + hiddenSize; j < nrUnits; j++){
			double in_j = BIAS*W[j][j]; /* unit input function for unit j = Sum(W_i,j * a_j) over i
			= sum of weights*activations (between unit j and units in previous layer) = vector w . vector x */
			for(int i = inputSize; i < inputSize + hiddenSize; i++){
				in_j += W[i][j]*a[i];
			}

			/* apply hard threshold function to unit input function for j (the vector component a_j, 
			 * given input vector "x", is 1 if the weight vector dot the input vector is >= 0, else 0 */
			if(in_j >= 0){
				ret[reti++] = 1;
			}
			else{
				ret[reti++] = 0;
			}
		}

		return ret;
	}

	/**The sigmoid activation function.
	 */
	double g(double x){
		return 1/ (1 + Math.exp(-1*x)); //= 1/ (1 + e^(-x))
	}
	
	/**The network is reborn.
	 */
	void randomRestart(){
		a = new double[nrUnits]; 
		W = new double[nrUnits][nrUnits];
		D = new double[nrUnits];
		//assign initial random weights
		for(int i = 0; i < W.length; i++){
			for(int j = 0; j < W[i].length; j++){
				if( (int)(Math.random()*2)/*[0,1]*/ == 1){
					W[i][j] = (int)(Math.random()*2); /*[0,2]*/
				}
				else{
					W[i][j] = -1*(int)(Math.random()*2); /*[-2,0]*/
				}
			}
		}
	}

	/**Redirect PrintStream so don't have to comment out everything for debugging.
	 * Source: http://stackoverflow.com/questions/8363493/hiding-system-out-print-calls-of-a-class
	 */
	void disablePrinting(){
		PrintStream theVoid = new PrintStream(new OutputStream(){
			public void write(int b) {
			}
		});
		System.setOut(theVoid);
	}

	/**Reenable printing.
	 */
	void enablePrinting(){
		System.setOut(ps);
	}
}