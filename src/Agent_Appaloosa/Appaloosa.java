package Agent_Appaloosa;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

import javax.xml.stream.events.EndDocument;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.issue.ValueDiscrete;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.classifiers.functions.LinearRegression;

/**
 * This is your negotiation party.
 */
public class Appaloosa extends AbstractNegotiationParty {

	private Logistic logistic_regressor = null;
	private Evaluation eval;
	private Bid lastReceivedBid = null;
	private Bid lastSendedBid = null;
	private double threshold;
	private double slack_threshold;
	SortedOutcomeSpace sos;
	private ArrayList<BidHolder> BidHolderList = new ArrayList<>();
	// private StandardInfoList history;
	private Bid Max = null;
	private Double reservationV;
	private Bid Min = null;
	private Range r = new Range(0, 0);
	private int NumberOfIssues;
	private int offercounter;
	ArrayList<Attribute> AttributeList = new ArrayList<>();
	Instances dataset;
	String opponentName;
	private int loopcounter;

	// AttributeList.add(new Attribute("Time"));
	// AttributeList.add(new Attribute("Utility"));

	@Override
	public void init(NegotiationInfo info) {

		super.init(info);

		sos = new SortedOutcomeSpace(utilitySpace);
		eval = null;
		offercounter = 0;
		opponentName = null;

		double time = timeline.getTime();

		try {

			Max = utilitySpace.getMaxUtilityBid();
			threshold = GetThreshold();
			slack_threshold = threshold;

			Min = utilitySpace.getMinUtilityBid();
			r.setUpperbound(getUtility(Max));
			reservationV = utilitySpace.getReservationValue();

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}



		String agentname = getPartyName(info.getAgentID().toString());

		NumberOfIssues = 0;

		NumberOfIssues = utilitySpace.getDomain().getIssues().size();

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		
		threshold = GetThreshold();

		if (logistic_regressor == null && timeline.getTime() > 0.60 && offercounter>0) {
			logistic_regressor = buildOpponentModel();
			BidHolderList = CreateBidsForModel();

			slack_threshold=threshold;
			
		}
		if (logistic_regressor != null) {
			if (threshold != slack_threshold) {


				BidHolderList = CreateBidsForModel();
	
				slack_threshold = threshold;

			}

			if (lastReceivedBid != null && getUtility(lastReceivedBid) >= threshold) {

				saveDataset();
				return new Accept(getPartyId(), lastReceivedBid);

			} else {

				int chooseBid = 0;

				if (Math.random() < 0.2) {
					chooseBid = 0;
				} else {
					chooseBid = getRandomNumberInRange(1, BidHolderList.size() - 1);
				}

				lastSendedBid = BidHolderList.get(chooseBid).bid;

				return new Offer(getPartyId(), lastSendedBid);

			}

		}

		else {

			if (lastReceivedBid != null && getUtility(lastReceivedBid) >= threshold) {
				saveDataset();

				System.out.println("Opponent's offer is accepted");

				return new Accept(getPartyId(), lastReceivedBid);

			} else {

				lastSendedBid = generateRandomBidWithUtility(threshold);
				return new Offer(getPartyId(), lastSendedBid);

			}

		}

	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);

		if (timeline.getTime() > 0.99)
			saveDataset();

		if (action instanceof Offer) {

			lastReceivedBid = ((Offer) action).getBid();

			if (opponentName == null) {

				opponentName = getPartyName(action.getAgent().toString());
			}

			if (offercounter == 0) {

				String filepath = getCurrentPath();
				filepath = filepath + "\\" + "\\" + getDescription() + "_dataof_" + opponentName + ".arff";

				if (DatasetExists(filepath)) {

					dataset = loadDataset();
					logistic_regressor = buildOpponentModel();

					BidHolderList = CreateBidsForModel();

					slack_threshold = threshold;


				} else {

					dataset = createDataset();

				}
				// deleteFile();
			}
			offercounter++;

			if (lastSendedBid != null) {
				addInstancetoDataset(lastSendedBid, "0");
				lastSendedBid = null;
			}

			addInstancetoDataset(lastReceivedBid, "1");

		} else if (action instanceof Accept) {


			System.out.println("My offer is accepted");
			saveDataset();

		} else if (action instanceof EndNegotiation) {

			saveDataset();

		}

	}

	private static int getRandomNumberInRange(int min, int max) {

		Random r = new Random();
		return r.ints(min, (max + 1)).findFirst().getAsInt();

	}

	private void addInstancetoDataset(Bid bid, String label) {

		int instance_index = dataset.numInstances();
		AbstractUtilitySpace utilitySpace = getUtilitySpace();
		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) utilitySpace;

		if (instance_index == -1)
			instance_index = 0;

		dataset.add(new DenseInstance(NumberOfIssues + 2));
		dataset.get(instance_index).setValue(0, timeline.getTime());

		for (int issue_index = 1; issue_index <= NumberOfIssues; issue_index++) {

			EvaluatorDiscrete evaluatorDiscrete = (EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issue_index);

			try {
				dataset.get(instance_index).setValue(issue_index,
						evaluatorDiscrete.getEvaluation((ValueDiscrete) bid.getValue(issue_index)));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}


		dataset.get(instance_index).setClassValue(label);

	}

	public double getAcceptanceProbability(Bid bid) {

		DenseInstance instance = new DenseInstance(NumberOfIssues + 2);
		AbstractUtilitySpace utilitySpace = getUtilitySpace();
		AdditiveUtilitySpace additiveUtilitySpace = (AdditiveUtilitySpace) utilitySpace;
		double acceptance_prob = 0;

		for (int issue_index = 1; issue_index <= NumberOfIssues; issue_index++) {

			EvaluatorDiscrete evaluatorDiscrete = (EvaluatorDiscrete) additiveUtilitySpace.getEvaluator(issue_index);

			try {
				instance.setValue(issue_index,
						evaluatorDiscrete.getEvaluation((ValueDiscrete) bid.getValue(issue_index)));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		instance.setValue(0, timeline.getTime());
		instance.setDataset(dataset);

		if (logistic_regressor != null) {
			try {
				eval = new Evaluation(dataset);
				eval.evaluateModelOnce(logistic_regressor, instance);

				acceptance_prob = logistic_regressor.distributionForInstance(instance)[1];

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return acceptance_prob;

	}

	private double GetThreshold() {

		double threshold = 0;
		double t = timeline.getTime();



		if (t <= 0.1) {
			threshold = 0.95;
		} else if (t < 0.35) {
			threshold = 0.90;
		} else if (t < 0.50) {
			threshold = 0.88;
		}  else if (t < 0.70) {
			threshold = 0.86;
		} else if (t < 0.80) {
			threshold = 0.85;
		}  else {
			threshold = 0.80;

		}

		return threshold;

	}

	private Logistic buildOpponentModel() {

		dataset.setClassIndex(dataset.numAttributes() - 1);
		Logistic logistic = new Logistic();

		try {
			logistic.buildClassifier(dataset);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

		}

		return logistic;

	}

	private String getCurrentPath() {

		String currentpath = "";

		String current = "";
		try {
			currentpath = new java.io.File(".").getCanonicalPath();
			String currentDir = System.getProperty("user.dir");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		return currentpath;
	}

	private String getPartyName(String partyID) {
		return partyID.substring(0, partyID.indexOf("@"));
	}

	private void saveDataset() {

		String filename = getCurrentPath();
		filename = filename + "\\" + getDescription() + "_dataof_" + opponentName + ".arff";

		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataset);
		try {
			saver.setFile(new File(filename));
			saver.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Save edilemedi");
			e.printStackTrace();
		}

	}

	private void deleteFile() {

		String filepath = getCurrentPath();
		filepath = filepath + "\\" + getDescription() + "_dataof_" + opponentName + ".arff";
		Path path = Paths.get(filepath);

		try {
			Files.delete(path);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private boolean DatasetExists(String filepath) {

		boolean exists = false;

		File f = new File(filepath);
		if (f.exists() && !f.isDirectory()) {
			exists = true;
		}

		return exists;
	}

	private Instances createDataset() {

		AttributeList.add(new Attribute("Time"));
		int i = 0;
		for (Issue issue : utilitySpace.getDomain().getIssues()) {
			i++;
			AttributeList.add(new Attribute("Issue " + i));

		}

		List<String> my_nominal_values = new ArrayList<String>();
		my_nominal_values.add("0");
		my_nominal_values.add("1");
		AttributeList.add(new Attribute("Reject / Accept", my_nominal_values));
		dataset = new Instances("Dataset", AttributeList, 2);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		return dataset;
	}

	private Instances loadDataset() {

		DataSource source = null;
		String filepath = getCurrentPath();
		filepath = filepath + "\\" + "\\" + getDescription() + "_dataof_" + opponentName + ".arff";

		try {

			source = new DataSource(filepath);
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes() - 1);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return dataset;
	}

	@Override
	public String getDescription() {
		return "Appaloosa";
	}

	public Bid generateRandomBidWithUtility(double utilityThreshold) {
		Bid randomBid;
		double utility;
		do {
			randomBid = generateRandomBid();
			try {
				utility = utilitySpace.getUtility(randomBid);
			} catch (Exception e) {
				utility = 0.0;
			}
		} while (utility < utilityThreshold);
		return randomBid;
	}

	public ArrayList<BidHolder> CreateBidsForModel() {

		r.setLowerbound(threshold);
		int upperbound = 0;
		List<BidDetails> bids = sos.getBidsinRange(r);
		PriorityQueue<BidHolder> BidHolderQueue = new PriorityQueue<BidHolder>();


		for (BidDetails bidTot : bids) {

			BidHolderQueue.add(new BidHolder(bidTot.getBid(), getAcceptanceProbability(bidTot.getBid())));
		}


		if (BidHolderQueue.size() >= 10) {
			upperbound = 10;
		} else {
			upperbound = BidHolderQueue.size();
		}

		BidHolderList = new ArrayList<BidHolder>();

		for (int i = 0; i < upperbound; i++) {

			BidHolderList.add(BidHolderQueue.poll());

		}

		return BidHolderList;
	}

	public class BidHolder implements Comparable<BidHolder> {
		public Bid bid;
		public Double probability;

		public BidHolder(Bid bid, double probability) {
			this.probability = probability;
			this.bid = bid;

		}

		public int compareTo(BidHolder arg0) {
			// TODO Auto-generated method stub
			if (arg0.probability - this.probability < 0)
				return -1;
			else if (this.probability - arg0.probability < 0)
				return 1;
			else
				return 0;
		}

	}

}
