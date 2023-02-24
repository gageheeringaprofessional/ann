import java.math.BigDecimal;
import java.math.RoundingMode;

/**@author Gage Heeringa
 * Completed 2/12/2016
 * CS 4811 - Program 1
 * 
 * Generate training examples for an XSquaredNetwork.
 */
public class TrainingSetGenerator {

	public static void main(String[] args){
		int c = 0;
		int range = 4; //x:[ -range, range )
		int total = 50*range;
		
		for(int i = 0; i < total; i++){
			double y, x = 0.0;
			int b = 0;

			if( (int)(Math.random()*2) == 1){ //50/50 chance
				x = round(Math.random()*range, 1); // [0, range] rounded to 1 decimal place
			}
			else{
				x = -1*round(Math.random()*range, 1); // [-range, 0] rounded to 1 decimal place
			}

			if( (int)(Math.random()*2) == 1){ //50/50 chance
				y = x*x + Math.random()*x; //y >= x^2
			}
			else{
				y = x*x - Math.random()*x; // y < x^2
			}
			y = round(y, 1);

			if(y >= x*x){
				c++;
				b = 1;
			}

			System.out.printf("new XSquaredExample(%.1f,%.1f, %d), ", y, x, b);
			if(Math.abs(i)%4 == 3)
				System.out.println();
		}

		System.out.printf("%d/%d had result 1", c, total);
	}

	/**Source: http://stackoverflow.com/questions/2808535/round-a-double-to-2-decimal-places
	 */
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();

		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(places, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}
}
