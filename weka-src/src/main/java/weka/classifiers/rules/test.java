/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ZeroR.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.rules;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;


//Ajouts pour CANC
import weka.core.OptionHandler;
import weka.filters.Filter;

/**
 * <!-- globalinfo-start --> Class for building and using a 0-R classifier.
 * Predicts the mean (for a numeric class) or the mode (for a nominal class).
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 12024 $
 */
public class test extends AbstractClassifier implements
  WeightedInstancesHandler, Sourcable {

  /** for serialization */
  static final long serialVersionUID = 48055541465867954L;

  /**
   * Un filtre permettant de transformer les données numériques en données nominales
   */

  //protected static Filter m_Filter = new weka.filters.unsupervised.attribute.Discretize();
  protected static Filter m_Filter = new weka.filters.supervised.attribute.Discretize();
  
  public void setFilter(Filter filter) {
    m_Filter = filter;
    }
  
  public Filter getFilter() {
    return m_Filter;     
    }
  
  protected String getFilterSpec() {
      
      Filter c = getFilter();
      if (c instanceof OptionHandler) {
          return c.getClass().getName() + " "
                  + Utils.joinOptions(((OptionHandler)c).getOptions());
      }
      return c.getClass().getName();
  }
  
  /** The instance structure of the filtered instances */
  protected Instances m_FilteredInstances;
  

  /** The class value 0R predicts. */
  private double m_ClassValue;

  /** The number of instances in each class (null if class numeric). */
  private double[] m_Counts;

  /** The class attribute. */
  private Attribute m_Class;

  /**
   * Returns a string describing classifier
   * 
   * @return a description suitable for displaying in the explorer/experimenter
   *         gui
   */
  public String globalInfo() {
    return "Class for building and using a 0-R classifier. Predicts the mean "
      + "(for a numeric class) or the mode (for a nominal class).";
  }
  
  /**
	 * L'apprentissage du concept nominal
	 */
  
	public static final int CONCEPT_LEARNING_FMAN = 1;  // Default: Fermeture de Meilleur Attribut Nominal
  
  private int NominalConceptLearning = CONCEPT_LEARNING_FMAN;
  
  public static final Tag [] TAGS_NominalConceptLearning = {
  	new Tag(CONCEPT_LEARNING_FMAN, "Closure of best nominal attribut"),
  	};
  
  public SelectedTag getConceptLearning() {	
  	return new SelectedTag(NominalConceptLearning, TAGS_NominalConceptLearning);	
  	}
  
  public void setConceptLearning(SelectedTag agregation) {
  	if (agregation.getTags() == TAGS_NominalConceptLearning)
  		this.NominalConceptLearning = agregation.getSelectedTag().getID();
  	}
  
  /**
   * Fermeture du Meilleur Attribut Nominal : choix de(s) valeur(s) nominale(s)
   */
  
  public static final int PM_GAIN_INFO = 1;	// Default: La valeur la plus pertinente (support) de l'attribut qui maximise le Gain Informationel
  public static final int PM_GAIN_RATIO = 2;	// Les valeurs nominales de l'attribut qui maximise LE GAIN RATIO
  public static final int PM_Correlation = 3; // les valeurs nominales qui  maximise le correlation attribut eval
  public static final int PM_HRATIO = 4;
  public static final int PM_InformationMutuelle = 5;
  public static final int PM_Symmetrical = 6; // les valeurs nominales qui maximise de symmetrical
  private int pertinenceMeasure = PM_GAIN_INFO;
  
  public static final Tag [] TAGS_pertinenceMeasure = {
			new Tag(PM_GAIN_INFO, "Gain Info"),
			new Tag(PM_GAIN_RATIO, "Gain Ratio"),
			new Tag(PM_Correlation,"Correlation"),
			new Tag(PM_HRATIO,"H-RATIO"),
			new Tag(PM_InformationMutuelle,"Information Mutuelle"),
			new Tag(PM_Symmetrical,"Symmetrical")};
	    
	    public SelectedTag getPertinence_Measure() {
	    	return new SelectedTag(pertinenceMeasure, TAGS_pertinenceMeasure);
			
			}

		public void setPertinence_Measure(SelectedTag agregation) {
			if (agregation.getTags() == TAGS_pertinenceMeasure)
				this.pertinenceMeasure = agregation.getSelectedTag().getID();
			}
		
		/**
	     * Le choix de la technique du vote 
	     * en cas o� nous avons retenu tout les valeurs nominales
	     * de l'attribut qui maximise le gain Informationel.
	     */
	    
	    public static final int Vote_Maj = 1;	// Vote majoritaire
	    public static final int Vote_Plur = 2;  // Vote pluralité
	        
	    private int VoteMethods = Vote_Maj;
	   
	    
	    public static final Tag [] TAGS_VoteMethods = {
			new Tag(Vote_Maj, "Majority Vote"),
			new Tag(Vote_Plur, "Plurality Vote"),
			};

	       
	    public SelectedTag getVote_Methods() {
			return new SelectedTag(VoteMethods, TAGS_VoteMethods);
			}

		public void setVote_Methods(SelectedTag agregation) {
			if (agregation.getTags() == TAGS_VoteMethods)
				this.VoteMethods = agregation.getSelectedTag().getID();
			}

  /**
   * Returns default capabilities of the classifier.
   * 
   * @return the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    //modif pour CANC
    result.enable(Capability.BINARY_ATTRIBUTES);
    //result.enable(Capability.DATE_ATTRIBUTES); //update by Nida
    //result.enable(Capability.STRING_ATTRIBUTES); //update by Nida
    //result.enable(Capability.RELATIONAL_ATTRIBUTES); //update by Nida
    //result.enable(Capability.MISSING_VALUES); //update by Nida

    // class
    result.enable(Capability.NOMINAL_CLASS);
    //result.enable(Capability.NUMERIC_CLASS); //update by Nida
    //result.enable(Capability.DATE_CLASS); //update by Nida
    //result.enable(Capability.MISSING_CLASS_VALUES); //update by Nida

    // instances
    result.setMinimumNumberInstances(0);

    return result;
  }


  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> result = new Vector<Option>();

    result.addElement(new Option(
      "\tFull class name of filter to use, followed\n"
        + "\tby filter options.\n"
        + "\teg: \"weka.filters.unsupervised.attribute.Remove -V -R 1,2\"\n"
        + "\t(default: weka.filters.MultiFilter with\n"
        + "\tweka.filters.unsupervised.attribute.ReplaceMissingValues)", "F",
      1, "-F <filter specification>"));

    return result.elements();
  }

  /**
   * Parses a given list of options.
   * <p/>
   * 
   * <!-- options-start --> Valid options are:
   * <p/>
   * 
   * <pre>
   * -F &lt;filter specification&gt;
   *  Full class name of filter to use, followed
   *  by filter options.
   *  eg: "weka.filters.unsupervised.attribute.Remove -V -R 1,2"
   *  (default: weka.filters.MultiFilter with
   *  weka.filters.unsupervised.attribute.ReplaceMissingValues)
   * </pre>
   * 
   * <pre>
   * -c &lt;the class index&gt;
   *  The class index. (default = last)
   * </pre>
   * 
   * <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    boolean  runString;
    
    //Ajout de pertinence Mesure
    
    runString = Utils.getFlag("PM_GAIN_INFO", options);
	if (runString)
		pertinenceMeasure = PM_GAIN_INFO;
	
	runString = Utils.getFlag("PM_GAIN_RATIO", options);
	if (runString)
		pertinenceMeasure = PM_GAIN_RATIO;
	
	runString = Utils.getFlag("PM_Correlation", options);
	if (runString)
		pertinenceMeasure = PM_Correlation;
	
	runString = Utils.getFlag("PM_HRATIO", options);
	if (runString)
		pertinenceMeasure = PM_HRATIO;
	
	runString = Utils.getFlag("PM_InformationMutuelle", options);
	if (runString)
		pertinenceMeasure = PM_InformationMutuelle;
	
	runString = Utils.getFlag("PM_Symmetrical", options);
	if (runString)
		pertinenceMeasure = PM_Symmetrical;
					
	switch (pertinenceMeasure) { 
	 case PM_GAIN_INFO                :	         pertinenceMeasure = 1; break;
	 case PM_GAIN_RATIO               :          pertinenceMeasure = 2; break;
	 case PM_Correlation              :          pertinenceMeasure = 3; break;
	 case PM_HRATIO                   :          pertinenceMeasure = 4; break;
	 case PM_InformationMutuelle      :          pertinenceMeasure = 5; break;
	 case PM_Symmetrical              :          pertinenceMeasure = 6; break;
     }
	
	// Les techniques de vote dans le cas de la fermeture des valeurs 
	// nominales de l'attribut qui m'aximise le gain informationel
	runString = Utils.getFlag("Vote_Maj", options);
	if ((pertinenceMeasure == PM_GAIN_INFO) && runString)
		VoteMethods = Vote_Maj;
	runString = Utils.getFlag("Vote_Plur", options);
	if ((pertinenceMeasure == PM_GAIN_INFO) && runString)
		VoteMethods = Vote_Plur;
						
	switch (VoteMethods) {
		case Vote_Plur  :	    VoteMethods = 1; break;
		case Vote_Maj   :		VoteMethods = 2; break;
	}	 
	
	String tmpStr;
    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() > 0) 
    {
      String[] filterSpec = Utils.splitOptions(tmpStr);
      if (filterSpec.length == 0)
        throw new IllegalArgumentException("Invalid filter specification string");
      String filterName = filterSpec[0];
      filterSpec[0] = "";
      setFilter((Filter) Utils.forName(Filter.class, filterName, filterSpec));
    } 
    else 
      setFilter(new weka.filters.supervised.attribute.Discretize());    
    
    m_Debug = Utils.getFlag('D', options);
    
    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Associator.
   * 
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    Vector<String> result = new Vector<String>();
    
    switch(NominalConceptLearning) 
	  {
	  case CONCEPT_LEARNING_FMAN: 	result.add("-fman"); break;
	  }

	  if(NominalConceptLearning == CONCEPT_LEARNING_FMAN)
		  switch(pertinenceMeasure) 
		  {
		  case PM_GAIN_INFO:	
			  result.add("-giMultiV");
			  switch(VoteMethods) 
			  {
			  case Vote_Maj:	result.add("-majVote"); break;
			  case Vote_Plur:	result.add("-plurVote"); break;
			  }
			  break;
			  
		  case PM_GAIN_RATIO:	
			  result.add("-giRatioBestV"); 
			  break;
			  
		  case PM_Correlation:
		      result.add("-giCorrelationBestV");break;
		      
		  case PM_HRATIO:
			  result.add("-giHRATIO");break;
			  
		  case PM_InformationMutuelle:
			  result.add("-giInformationMutuelle");break;
		  
		  case PM_Symmetrical:
		      result.add("-giSymmetricalBestV");break;
		  }
    
    result.add("-F");
    result.add("" + getFilterSpec());
    
    if (m_Debug)
        result.add("-D");

    Collections.addAll(result, super.getOptions());

    return result.toArray(new String[result.size()]);
  }

  /**
   * Generates the classifier.
   * 
   * @param instances set of instances serving as training data
   * @throws Exception if the classifier has not been generated successfully
   */
  @Override
  public void buildClassifier(Instances instances) throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    double sumOfWeights = 0;

    m_Class = instances.classAttribute();
    m_ClassValue = 0;
    switch (instances.classAttribute().type()) {
    case Attribute.NUMERIC:
      m_Counts = null;
      break;
    case Attribute.NOMINAL:
      m_Counts = new double[instances.numClasses()];
      for (int i = 0; i < m_Counts.length; i++) {
        m_Counts[i] = 1;
      }
      sumOfWeights = instances.numClasses();
      break;
    }
    for (Instance instance : instances) {
      double classValue = instance.classValue();
      if (!Utils.isMissingValue(classValue)) {
        if (instances.classAttribute().isNominal()) {
          m_Counts[(int) classValue] += instance.weight();
        } else {
          m_ClassValue += instance.weight() * classValue;
        }
        sumOfWeights += instance.weight();
      }
    }
    if (instances.classAttribute().isNumeric()) {
      if (Utils.gr(sumOfWeights, 0)) {
        m_ClassValue /= sumOfWeights;
      }
    } else {
      m_ClassValue = Utils.maxIndex(m_Counts);
      Utils.normalize(m_Counts, sumOfWeights);
    }
  }

  /**
   * Classifies a given instance.
   * 
   * @param instance the instance to be classified
   * @return index of the predicted class
   */
  @Override
  public double classifyInstance(Instance instance) {

    return m_ClassValue;
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   * 
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if class is numeric
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    if (m_Counts == null) {
      double[] result = new double[1];
      result[0] = m_ClassValue;
      return result;
    } else {
      return m_Counts.clone();
    }
  }

  /**
   * Returns a string that describes the classifier as source. The classifier
   * will be contained in a class with the given name (there may be auxiliary
   * classes), and will contain a method with the signature:
   * 
   * <pre>
   * <code>
   * public static double classify(Object[] i);
   * </code>
   * </pre>
   * 
   * where the array <code>i</code> contains elements that are either Double,
   * String, with missing values represented as null. The generated code is
   * public domain and comes with no warranty.
   * 
   * @param className the name that should be given to the source class.
   * @return the object source described by a string
   * @throws Exception if the souce can't be computed
   */
  @Override
  public String toSource(String className) throws Exception {
    StringBuffer result;

    result = new StringBuffer();

    result.append("class " + className + " {\n");
    result.append("  public static double classify(Object[] i) {\n");
    if (m_Counts != null) {
      result.append("    // always predicts label '"
        + m_Class.value((int) m_ClassValue) + "'\n");
    }
    result.append("    return " + m_ClassValue + ";\n");
    result.append("  }\n");
    result.append("}\n");

    return result.toString();
  }

  /**
   * Returns a description of the classifier.
   * 
   * @return a description of the classifier as a string.
   */
  @Override
  public String toString() {

    if (m_Class == null) {
      return "test: No model built yet.";
    }
    if (m_Counts == null) {
      return "test predicts class value: " + m_ClassValue;
    } else {
      return "test predicts class value: " + m_Class.value((int) m_ClassValue);
    }
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 12024 $");
  }

  /**
   * Main method for testing this class.
   * 
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new test(), argv);
  }
}
