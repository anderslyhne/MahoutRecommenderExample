package mahout_recommendersystem;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * Simple Mahout recommender system example with performance evaluation.
 * Based on <a href="https://mahout.apache.org/users/recommender/userbased-5-minutes.html">
 * https://mahout.apache.org/users/recommender/userbased-5-minutes.html</a>, but complete and commented, 
 * and it compiles.
 * <p>
 * The dataset is in "dataset.csv" and has the following format:
 * <p>
 * <i>userID,itemID,value</i>
 * <p>
 * <i>userID</i> represents the user who interacted with an item with id <i>itemID</i>. 
 * The field <i>value</i> corresponds to the user's rating of the item (or the strength 
 * of the interaction). Except from dataset.csv:
 * <p>
 * <pre>
 * 1,10,1.0
 * 1,11,2.0
 * 1,12,5.0
 * 1,13,5.0
 * 1,14,5.0
 * 1,15,4.0
 * 1,16,5.0
 * ...
 * </pre>
 *  
 *  <p>
 *  I used Mahout 0.13.0. JavaDoc: <a href="https://mahout.apache.org/docs/0.13.1-SNAPSHOT/javadocs/index.html">https://mahout.apache.org/docs/0.13.1-SNAPSHOT/javadocs/index.html</a>s
 *  </p>
 *  
 * @author alc
 */
public class MahoutRecommenderExample implements RecommenderBuilder {
	private DataModel fullDataModel;
	
	/**
	 * Create the object and load the full data model from "dataset.csv".
	 * @throws IOException
	 */
	public MahoutRecommenderExample() throws IOException {
		this.fullDataModel = new FileDataModel(new File("dataset.csv"));
	}

	/**
	 * Create a simple recommender that uses <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient">
	 * Pearson's correlation</a> as the similarity measure between users. The data model is provided as argument to be able 
	 * to evaluate/score the model.
	 * @return A PCC similarity, threshold-based neighborhood recommender 
	 * @throws TasteException 
	 */
	public Recommender buildRecommender(DataModel dm) throws TasteException {
		UserSimilarity similarity     = new PearsonCorrelationSimilarity(dm);
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, dm);
		return new GenericUserBasedRecommender(dm, neighborhood, similarity);
	}

	/**
	 * Get a recommender for the full data model
	 * @return
	 * @throws TasteException
	 */
	public UserBasedRecommender getRecommender() throws TasteException {
		return (UserBasedRecommender) buildRecommender(fullDataModel); 
	}
	
	/**
	 * Evaluate the recommender. Use 90% for training and 100% for evaluation.
	 * 
	 * @return problem-dependent score reflecting performance (0 = perfect)
	 * @throws TasteException
	 */
	public double scoreRecommender() throws TasteException {
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		double result = evaluator.evaluate(this, null, fullDataModel, 0.9, 1.0);
		return result;
	}
	
	public static void main(String[] args) throws IOException, TasteException {
		MahoutRecommenderExample example = new MahoutRecommenderExample();
		
		// Get recommender based on full model
		UserBasedRecommender recommender = example.getRecommender();
		
		// For user 2, recommend 3 items 
		List<RecommendedItem> recommendations = (List<RecommendedItem>) recommender.recommend(2, 3);

		System.out.println("Recommendations for user #2:");
		for (RecommendedItem recommendation : recommendations) {
		  System.out.println("\t" + recommendation);
		}

		// Scoring 10 times as the evaluation is stochastic. Also, computations sometimes lead to NaN.
		for (int i = 0; i< 10; i++)
			System.out.println("Score " + i + ": " + example.scoreRecommender());
	}
}
