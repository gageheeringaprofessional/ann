***** How to run your program with the training sets you provided
Given XSquaredNetwork.java, MultilayerPerceptron.java, and TrainingSetGenerator.java, use Eclipse and simply run.

Alternatively: 
1) Open the command prompt and go to the directory containing XSquaredNetwork.java, MultilayerPerceptron.java, and TrainingSetGenerator.java
2) Compile: javac XSquaredNetwork.java MultilayerPerceptron.java TrainingSetGenerator.java
3) Run: java XSquaredNetwork


***** How to change the training sets
In XSquaredNetwork.java, go to the main method.  The default training sets are defined as:	
	XSquaredExample[] examples = new XSquaredExample[]{ ... };
	
You may edit these training sets and add others.
To generate random tests, you may use the TrainingSetGenerator class.


***** How to graph the results
In XSquaredNetwork.java, go to the main method.  Uncomment the following line at the end of main:
			//x.generatePoints();

After learning, the method generatePoints() creates two files:
"generatedGEQPoints.txt" contains coordinates such that y >= x^2.
"generatedLPoints.txt" contains coordinates such that y < x^2.

Procedure to make the graph using Microsoft Excel:
Open Excel 
-> Highlight columns A and B 
-> go to the DATA tab 
-> Click From Text 
-> Import generatedGEQPoints.txt 
-> Select "Delimited" -> Click Next 
-> Insert a comma , in "Other:" -> Click Next -> Click Finish -> Click OK.

Highlight columns C and D 
-> go to the DATA tab 
-> Click From Text 
-> Import generatedLPoints.txt 
-> Select "Delimited" -> Click Next 
-> Insert a comma , in "Other:" -> Click Next -> Click Finish -> Click OK

There are likely to be more "y >= x^2" points (columns A(x),B(y)) than "<" points (columns C(x),D(y)), or vice-versa.
If there are more rows in A and B than C and D:
	Fill the remaining entries in columns (A,B) with (0,0) (0 >= 0^2) so that columns A,B,C, and D all have the same number of entries.
Else if there are more rows in C and D than A and B:
	Fill the remaining entries in columns (C,D) with (1,0) (0 < 1^1) so that columns A,B,C, and D all have the same number of entries.
	Note: for my graph, I had to fill columns C,D for rows 4657-9824
	
Now highlight columns A and B.
->Go to the INSERT tab
->Scatter (plot)
->Right click the scatter plot
->Click "Select data..."
->Click "Add" under "Legend Entries (Series)"
->Enter series name as "y < x^2" or something
->Enter X values as column C (=Sheet1!$C:$C)
->Enter Y values as column D (=Sheet1!$D:$D)
->Click OK twice


***** How to change the network structure
In XSquaredNetwork.java, go to the main method.  In the first line, the network is created:
			XSquaredNetwork x = new XSquaredNetwork(2, 5, 1);
			
The first, second, and third parameters of the constructor specify the number of units in the input, hidden, and output layers of the network respectively.

Note that XSquaredNetwork is a subclass of MultilayerPerceptron, so the number of units for the input and output layers is already set to 2 and 1 respectively, as the y >= x^2 function always has 2 inputs (x and y) and 1 output (true or false).  Only the number of units in the hidden layer should be tampered with.