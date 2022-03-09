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


import weka.attributeSelection.*;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.filters.unsupervised.attribute.ReplaceMissingValues; //Prit sur le code de Marwa
import weka.classifiers.Classifier; //Pour buildClassifier

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.fca.Classification_Rule;
import weka.classifiers.fca.Classify_Instance;
//import weka.classifiers.fca.HRatio;
//import weka.classifiers.fca.InformationMutuelle;
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
public class testConceptLearning extends AbstractClassifier implements
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
  
  //Pour buildClassifier
  protected ReplaceMissingValues m_Missing = new ReplaceMissingValues();
  protected int m_CNC = 0;
  protected Classifier m_ZeroR;
  Calendar calendar;
  SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");
  
  //MODÈLE
  
  /**
	 * L'apprentissage du concept nominal
	 */
  
	public static final int CONCEPT_LEARNING_FMAN_BESTV = 1;
	public static final int CONCEPT_LEARNING_FMAN_MULTIV = 2;
	public static final int CONCEPT_LEARNING_FTAN_BESTV = 3;
	public static final int CONCEPT_LEARNING_FTAN_MULTIV = 4;
  
  private int NominalConceptLearning = CONCEPT_LEARNING_FMAN_BESTV;
  
  public static final Tag [] TAGS_NominalConceptLearning = {
  	new Tag(CONCEPT_LEARNING_FMAN_BESTV, "C BV of PA"), // -BestV et -fman
  	new Tag(CONCEPT_LEARNING_FMAN_MULTIV, "C AV of PA"), // -MultiV et -fman
  	new Tag(CONCEPT_LEARNING_FTAN_BESTV, "C BV of Aa"), // -BestV et -ftan
  	new Tag(CONCEPT_LEARNING_FTAN_MULTIV, "C AV of Aa") // -MultiV et -ftan
  	};
  
  public SelectedTag getConceptLearning() {	
  	return new SelectedTag(NominalConceptLearning, TAGS_NominalConceptLearning);	
  	}
  
  public void setConceptLearning(SelectedTag agregation) {
  	if (agregation.getTags() == TAGS_NominalConceptLearning)
  		this.NominalConceptLearning = agregation.getSelectedTag().getID();
  	}
  

  //Classify Instance
  public static ArrayList <Classification_Rule> m_classifierNC;
  
  protected static InfoGainAttributeEval  m_InfoGainAttributeEval =  new InfoGainAttributeEval();
  protected static GainRatioAttributeEval m_GainRatioAttributeEval = new GainRatioAttributeEval();
  //protected static InformationMutuelle m_InformationMutuelle = new InformationMutuelle();
  //protected static HRatio m_HRatio = new HRatio();
  //protected static OneRAttributeEval      m_OneRAttributeEval      = new OneRAttributeEval();
  protected static CorrelationAttributeEval  m_CorrelationAttributeEval = new CorrelationAttributeEval();
  protected static SymmetricalUncertAttributeEval  m_SymmetricalUncertAttributeEval = new SymmetricalUncertAttributeEval();
  
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
			new Tag(PM_GAIN_INFO, "G Inf"),
			new Tag(PM_GAIN_RATIO, "G Rat"),
			new Tag(PM_Correlation,"Corr"),
			new Tag(PM_HRATIO,"H-RATIO"),
			new Tag(PM_InformationMutuelle,"Inf Mut"),
			new Tag(PM_Symmetrical,"Symm")};
	    
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
			new Tag(Vote_Maj, "MajV"),
			new Tag(Vote_Plur, "PluralV"),
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
    
 // Fermeture du meilleur attribut nominal	
 			runString = Utils.getFlag("CONCEPT_LEARNING_FMAN_BESTV", options);
 			if (runString)
 				NominalConceptLearning = CONCEPT_LEARNING_FMAN_BESTV; // Experimental
 			
 			switch (NominalConceptLearning) { 
 			case CONCEPT_LEARNING_FMAN_BESTV    :    NominalConceptLearning = 1; break;
 			case CONCEPT_LEARNING_FMAN_MULTIV   :    NominalConceptLearning = 2; break;
 			case CONCEPT_LEARNING_FTAN_BESTV    :    NominalConceptLearning = 3; break;
 			case CONCEPT_LEARNING_FTAN_MULTIV   :    NominalConceptLearning = 4; break;
 			}
    
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
	  case CONCEPT_LEARNING_FMAN_BESTV: 	result.add("-fman"); break;
	  case CONCEPT_LEARNING_FMAN_MULTIV:    result.add("-fman"); break;
	  case CONCEPT_LEARNING_FTAN_BESTV:     result.add("-ftan"); result.add("-BestV"); break;
	  case CONCEPT_LEARNING_FTAN_MULTIV:    result.add("-ftan"); result.add("-MultiV"); break;
	  }
    
    if(NominalConceptLearning == CONCEPT_LEARNING_FMAN_BESTV) {
    	switch(pertinenceMeasure)
  	  {
  		  case PM_GAIN_INFO:
  			  result.add("-giBestV"); break;	
  			  /*result.add("-giBestV");
  			  switch(VoteMethods) 
  			  {
  			  case Vote_Maj:	result.add("-majVote"); break;
  			  case Vote_Plur:	result.add("-plurVote"); break;
  			  }
  			  break;*/
  			  
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
    }
    
    if(NominalConceptLearning == CONCEPT_LEARNING_FMAN_MULTIV) {
    	result.add("-giMultiV");
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
public void buildClassifier(Instances instances) throws Exception {
      
      // can classifier handle the data?
      getCapabilities().testWithFail(instances);
    
      // remove instances with missing class
      m_FilteredInstances = new Instances(instances);
      m_FilteredInstances.deleteWithMissingClass();
      
      if (m_FilteredInstances.numInstances() == 0)
          throw new Exception("No training instances left after removing instances with MissingClass!");
      
      /*
      m_Missing = new ReplaceMissingValues();
      m_Missing.setInputFormat(instances);
      instances = Filter.useFilter(instances, m_Missing); 
      */
      m_Missing.setInputFormat(m_FilteredInstances);
      m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Missing);
      
      if (m_FilteredInstances.numInstances() == 0)
          throw new Exception("No training instances left after removing instances with MissingValues!");
           
      m_Filter.setInputFormat(m_FilteredInstances);  // filter capabilities are checked here
      m_FilteredInstances = Filter.useFilter(m_FilteredInstances, m_Filter);

        
      // only class? -> build ZeroR model!! si on un seul attribut on appelle le classifieur zeroR
      if (m_FilteredInstances.numAttributes() == 1) 
      {
          System.err.println("Cannot build model (only class attribute present in data!), "
           + "using ZeroR model instead!");
          m_ZeroR = new weka.classifiers.rules.ZeroR();
          m_ZeroR.buildClassifier(m_FilteredInstances);
          return;
      } 
      else {
        m_ZeroR = null;
        this.m_CNC = 1; // build CNC model
        }
        
      
      
      switch(this.NominalConceptLearning)
      {
      case CONCEPT_LEARNING_FMAN_BESTV: 
          if(m_Debug){
              calendar = Calendar.getInstance();
              System.out.println("\n \t"+sdf.format(calendar.getTime()));  
          }
          //for (int i=0; i<m_FilteredInstances.numInstances();i++)
              //System.out.println(m_FilteredInstances.instance(i).toString());
          buildClassifierWithNominalClosure(m_FilteredInstances);  
          break;
      }
  }

protected void buildClassifierWithNominalClosure(Instances LearningData) throws Exception {

	  m_classifierNC = new ArrayList <Classification_Rule> ();	
	  m_classifierNC.clear();
	  
	  switch (this.pertinenceMeasure) 
	  {
	  case PM_GAIN_INFO: // Fermeture de la valeur nominale la plus pertienente (Support) de l'attribut nominal qui maximise le Gain Informationel 
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 1);
		  break; 
		
	  case PM_GAIN_RATIO: // Fermeture du Meilleur Attribut Nominal selon les classes
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 2);
		  break;
		    
	  case PM_Correlation: // Fermetures des valeurs nominales de l'attribut nominal qui maximise le Gain Informatioonel
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 3);
		  break;
	  
	  /*case PM_HRATIO: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE GAIN RATIO
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 4);
		  break;
		  
	  case PM_InformationMutuelle: // Fermetures des valeurs nominales de l'attribut nominal qui maximise LE ONE R
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 5);
		  break;
		  
	  case PM_Symmetrical: // Fermetures des valeurs nominales de l'attribut nominal qui maximise la correlation
		  m_classifierNC = ExtraireRegleFermNom(LearningData, 6);
		  break;*/
	  }
	  
	  if(m_Debug)	{
		  System.out.println("\n\n\t=== Vector CLASSIFIER NOMINAL CONCEPT ===");
		  for(int i=0; i<m_classifierNC.size();i++)
			  System.out.println("CNC["+i+"]: "+m_classifierNC.get(i).affich_nom_rule(true));
		}
}  
  

  /**
   * Classifies a given instance.
   * 
   * @param instance the instance to be classified
   * @return index of the predicted class
   */
  @Override
  public double classifyInstance(Instance inst) throws Exception  {
	  // default model?
	  /*if (m_ZeroR != null) 
		  return m_ZeroR.classifyInstance(inst);
	  */
	  
    //System.err.println("FilteredClassifier:: " + m_Filter.getClass().getName() + " in: " + inst);

    if (m_Filter.numPendingOutput() > 0) {
      throw new Exception("Filter output queue not empty!");
    }
    
    if (!m_Filter.input(inst)) {
      throw new Exception("Filter didn't make the test instance immediately available!");
    }
    m_Filter.batchFinished();
    Instance newInstance = m_Filter.output();

    //System.err.println("FilteredClassifier:: " + m_Filter.getClass().getName() + " out: " + newInstance); 
    
    m_Missing.input(inst);
    m_Missing.batchFinished();
    inst = m_Missing.output();

	  double result= (double) -1.0;
	  Classify_Instance  listRules = new Classify_Instance();
	  switch(pertinenceMeasure)
	  {
	  case PM_GAIN_RATIO : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case PM_Correlation : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case PM_HRATIO : 
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case PM_InformationMutuelle:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case PM_Symmetrical:
		  result = (double) listRules.classify_Instance_nom(newInstance, (Classification_Rule) m_classifierNC.get(0)); 
		  break;
	  case PM_GAIN_INFO: 
		  switch(VoteMethods)
		  {
		  case Vote_Plur: result = (double) listRules.classify_Instance_nom_VotePond(newInstance, m_classifierNC,newInstance.numClasses()); break;
		  case Vote_Maj: result = (double) listRules.classify_Instance_nom_VoteMaj(newInstance, m_classifierNC,newInstance.numClasses()); break;
		  }
		  break;
		  
	  }
	  
	  if (result == -1.0) 		
		  return Utils.missingValue();
	  
	  return result;
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
    runClassifier(new testConceptLearning(), argv);
  }
  
  public ArrayList<Classification_Rule> ExtraireRegleFermNom(Instances inst, int critere) throws Exception {
	  
		ArrayList <Classification_Rule> classifierNC= new ArrayList<Classification_Rule>();
				
		if(m_Debug)
		{
			System.out.println("\nAffichage du context non binaire");
			System.out.println("\tListe des attributs:");
			System.out.print("\t");
			for(int i=0; i<inst.numAttributes();i++)
				System.out.print("("+(i+1)+")"+inst.attribute(i).name()+"  ");
			System.out.println("\n\tContext:");
			for (int i=0 ; i<inst.numInstances(); i++)
				//System.out.println("\t"+(i+1)+" : "+inst.instance(i).toString());
				System.out.println("\t"+inst.instance(i).toString());
		}
		
		m_InfoGainAttributeEval.buildEvaluator(inst);
		
		// Compute attribute with maximum information gain (FROM ID3).
	    double[] infoGains = new double[inst.numAttributes()];
	    Enumeration attEnum = inst.enumerateAttributes();
	    while (attEnum.hasMoreElements()) {
	      Attribute att = (Attribute) attEnum.nextElement();
	      //infoGains[att.index()] = computeInfoGain(inst, att);
	      infoGains[att.index()] = m_InfoGainAttributeEval.evaluateAttribute(att.index());
		    }   
	    
	    if(m_Debug){
	    	System.out.println("\nCalcul des gains informationels de chaque attribut de ce context");
		    for(int i=0; i<inst.numAttributes();i++)
		    	System.out.println("\tInfoGains de l'attribut "+inst.attribute(i).name()+": "+infoGains[i]);
	    }
	    
	    Attribute m_Attribute;
	    m_Attribute = inst.attribute(Utils.maxIndex(infoGains));
	    if(m_Debug){	
	    	System.out.println("\nL'attribut retenu pour calculer sa fermeture: "+m_Attribute.name());
	    	System.out.println("\tAttribut d'indice "+m_Attribute.index());
	    	System.out.println("\tNombre des differents valeurs possibles: "+inst.numDistinctValues(m_Attribute.index()));
	    	for(int i=0; i<inst.numDistinctValues(m_Attribute.index()); i++)
	    		System.out.println("\t\tValeur "+(i+1)+" : "+inst.attribute(m_Attribute.index()).value(i));
	    }
		
		/* G�n�ration d'un classifieur de type CNC � partir de la fermeture 
	     * de la valeur nominale la plus pertinente (qui maximise le Support) 
	     * de l'attribut nominal qui maximise le Gain Informationel
	     */
	    if(critere == 1) 
	    {
	    	if(m_Debug)
	    		System.out.println("\nG�n�ration d'un CNC � partir de la fermeture du valeur la plus pertinente de l'attribut retenu");
			
	    	int supportDistVal=0;
			int indexBestDistVal = 0;
			int suppBestDistVal = 0;

			//Parcourir les differentes valeurs du 'm_Attribute'  
			for(int i=0; i<inst.numDistinctValues(m_Attribute.index()); i++)
			{				
				//Calcul du support de cette DistinctValue
				ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
				instDistVal.clear();
				
				supportDistVal=0;
				for(int j=0; j<inst.numInstances(); j++)
				{
					//System.out.print((j+1)+"i�me instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(i));
					if( inst.instance(j).stringValue(m_Attribute.index()) == inst.attribute(m_Attribute.index()).value(i))
					{
						supportDistVal++;
						instDistVal.add(j);
						//System.out.println("     OK");					
					}
					//else
						//System.out.println("     NO");
				}
				//System.out.println("Support de cette DistinctValue ("+inst.attribute(m_Attribute.index()).value(i)+"): "+supportDistVal);
				if(suppBestDistVal <= supportDistVal)
				{
					suppBestDistVal=supportDistVal;
					indexBestDistVal = i;
				}
			}	
			
			if(m_Debug)
				System.out.println("Meilleur DistinctValue: ( "+m_Attribute.value(indexBestDistVal)+" ) avec un support qui vaut: "+suppBestDistVal);
			
			//Extraires les exemples associ�s a cet attribut
			ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
			ArrayList <String> FERM_att = new ArrayList <String> ();	
	
			//Liste des instances verifiant la fermeture
			for(int i=0; i<inst.numInstances(); i++)
			{
				if( inst.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indexBestDistVal))
					FERM_exe.add(i);
			}
			int nbrInstFer = FERM_exe.size();
			
			if(m_Debug)
			{
				System.out.print("Fermeture des instances: \n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
				for(int i=0; i<nbrInstFer; i++)
					System.out.print(FERM_exe.get(i)+" - ");
				System.out.println();
				
				for(int i=0; i<nbrInstFer; i++)
					System.out.println("\t\t"+FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
			}
			
			//Liste des attributs associ�s � la fermeture 
			String nl= "-null-";
			for(int i=0; i<(int) inst.numAttributes()-1;i++)
			{
				int cmpt=0;
				String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
				//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
				for (int j=0; j<nbrInstFer;)
				{
					if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
					{
						cmpt++;
						if(cmpt==nbrInstFer)
						{
							FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
							//System.out.println(" ---> ok ");
						}
						j++;
					}
					else
					{
						j=nbrInstFer;
						FERM_att.add(nl); 
						//System.out.println(" ---> null ");
					}
				}
			}
			
			if(m_Debug)
			{
				System.out.print("Liste des attributs nominative:         ");
				for(int i=0; i<inst.numAttributes()-1;i++)
					System.out.print(inst.attribute(i).name()+" , ");
				System.out.print("\nListe des valeurs d'attribut retenues:  ");
				for(int i=0; i<FERM_att.size(); i++)
					System.out.print(FERM_att.get(i)+" , ");
			}	
			
			///////////////Extraire la classe majoritaire associ�e////////////////////		
			int [] nbClasse = new int [inst.numClasses()];			
			for(int k=0 ; k<inst.numClasses() ; k++)
				nbClasse[k]=0;
				
			//Parcourir les exemples associ�e � ce concept
			for(int j=0;j<nbrInstFer;j++)
				nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
				
			//Detertminer l'indice de la classe associ�e
			int indiceMax=0;
			for(int i=0;i<inst.numClasses();i++)
				if ( nbClasse[i] > nbClasse[indiceMax] )
					indiceMax=i;			
			if(m_Debug)
				System.out.println ("\nLa Classe Majoritaire est: "+inst.attribute(inst.classIndex()).value(indiceMax));
			
			// On retourne le concept Pertinent comme un vecteur de String 
			ArrayList <String> CP= new ArrayList <String>();
			for (int i=0 ; i<(int) inst.numAttributes()-1 ; i++)
					CP.add(FERM_att.get(i));
			
			Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax),infoGains[Utils.maxIndex(infoGains)]);
			classifierNC.add(r);
			
	    }
	    
	    
	    /* G�n�ration d'un classifieur de type CNC � partir de la fermeture 
	     * de la valeur nominale la plus pertienente (qui maximise le Support) 
	     * de l'attribut nominal qui maximise le Gain Informationel
	     */
	    //G�n�ration d'un classifieur faible � partir de la fermeture du meilleur attribut retenu 'm_Attribute'
	    /*
	    if(critere == 2)	// FMAN_GAIN_INFO_BA: Fermeture du Meilleur Attribut Nominal selon les classes
	    {
	    	int supportDistVal=0;
			int indexBestDistVal = 0;
			int suppBestDistVal = 0;

			for(int cl=0; cl<inst.numClasses(); cl++)
			{
				supportDistVal=0;
				indexBestDistVal = 0;
				suppBestDistVal = 0;
				
					//Extraction des indices d'instances etiquit�es par la classe d'indice <cl>
					ArrayList <Integer> IndTrainingbyClass = new ArrayList <Integer>();
					IndTrainingbyClass.clear();
					for(int i=0; i<inst.numInstances();i++)
						if(inst.instance(i).classValue()==cl)
							IndTrainingbyClass.add(i);
					
					//Extraction de l'�chantillon d'instances etiquit�es par la classe d'indice <cl>
					Instances TrainingbyClass = new Instances( inst, 0,IndTrainingbyClass.size()); 
					TrainingbyClass.delete();
				    for (int h=0; h< IndTrainingbyClass.size(); h++)
				    	TrainingbyClass.add(inst.instance(IndTrainingbyClass.get(h)));
				    if(TrainingbyClass.numInstances()==0)
				       	System.out.println("\nCAS PARTICULIER: TrainingbyClass.numInstances()==0");
				    
				    if(TrainingbyClass.numInstances()!=0)
				    {
						for(int i=0; i<TrainingbyClass.numDistinctValues(m_Attribute.index()); i++)
						{
							//Calcule du support de cette DistinctValue
							ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
							instDistVal.clear();
							
							supportDistVal=0;
							for(int j=0; j<TrainingbyClass.numInstances(); j++)
							{
								//System.out.print((j+1)+"i�me instance: "+TrainingbyClass.instance(j).stringValue(m_Attribute.index()) +" - "+ TrainingbyClass.attribute(m_Attribute.index()).value(i));
								if( TrainingbyClass.instance(j).stringValue(m_Attribute.index()) == TrainingbyClass.attribute(m_Attribute.index()).value(i))
								{
									supportDistVal++;
									instDistVal.add(j);
									//System.out.println("     OK");					
								}
								//else
									//System.out.println("     NO");
							}
							//System.out.println("Support de cette DistinctValue ("+TrainingbyClass.attribute(m_Attribute.index()).value(i)+"): "+supportDistVal);
							if(suppBestDistVal <= supportDistVal)
							{
								suppBestDistVal=supportDistVal;
								indexBestDistVal = i;
							}
						}	
						
						if(m_Debug)
							System.out.println("Indice du meilleur DistinctValue("+m_Attribute.value(indexBestDistVal)+"): "+indexBestDistVal+" avec un support qui vaut: "+suppBestDistVal);
						
						//Extraires les exemples associ�s a cet attribut
						ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
						ArrayList <String> FERM_att = new ArrayList <String> ();	
				
						//Liste des instances verifiant la fermeture
						for(int i=0; i<TrainingbyClass.numInstances(); i++)
						{
							if( TrainingbyClass.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indexBestDistVal))
								FERM_exe.add(i);
						}
						int nbrInstFer = FERM_exe.size();

						if(m_Debug)
						{
							System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
							for(int i=0; i<nbrInstFer; i++)
								System.out.print (" - "+FERM_exe.get(i));
							System.out.println();
							for(int i=0; i<nbrInstFer; i++)
								System.out.println(FERM_exe.get(i)+" : "+TrainingbyClass.instance(FERM_exe.get(i)).toString());
						}
						
						//Liste des attributs associ�s � la fermeture ??????????????????????
						//System.out.println("Extraction des attributs associ�s � cette fermeture");
						String nl= "-null-";
						for(int i=0; i< (int) inst.numAttributes()-1;i++)
						{
							int cmpt=0;
							String FirstDistVal = TrainingbyClass.instance(FERM_exe.get(0)).stringValue(i);
							//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
							for (int j=0; j<nbrInstFer;)
							{
								if(TrainingbyClass.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
								{
									cmpt++;
									if(cmpt==nbrInstFer)
									{
										FERM_att.add(TrainingbyClass.instance(FERM_exe.get(0)).stringValue(i));
										//System.out.println(" ---> ok ");
									}
									j++;
								}
								else
								{
									j=nbrInstFer;
									FERM_att.add(nl); 
									//System.out.println(" ---> null ");
								}
							}
						}
						
						if(m_Debug)
						{
						System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
							for(int i=0; i<FERM_att.size(); i++)
								System.out.print (FERM_att.get(i)+" , ");
						}
													
						//On retourne le concept Pertinent comme un vecteur 
						ArrayList <String> CP= new ArrayList <String>();
						for (int i=0; i< (int) inst.numAttributes()-1;i++)
								CP.add(FERM_att.get(i));
						
						Classification_Rule r = new Classification_Rule(CP,cl,inst.attribute(inst.classIndex()).value(cl));
						classifierNC.add(r);
				    }
			}
	    }
	    */

	    /* G�n�ration d'un classifieur de type CNC � partir de la fermeture 
	     * des valeurs nominales de l'attribut nominal qui maximise le Gain Informationel
	     */
	    /*
	    if(critere == 3)
	    {
	    	ArrayList <Integer> instDistVal =new ArrayList <Integer> ();
	    	
			for(int indDistVal=0; indDistVal<inst.numDistinctValues(m_Attribute.index()); indDistVal++)
			{
				instDistVal.clear();
				
				for(int j=0; j<inst.numInstances(); j++)
				{
					//System.out.print((j+1)+"i�me instance: "+inst.instance(j).stringValue(m_Attribute.index()) +" - "+ inst.attribute(m_Attribute.index()).value(indDistVal));
					if( inst.instance(j).stringValue(m_Attribute.index()) == inst.attribute(m_Attribute.index()).value(indDistVal))
					{
						instDistVal.add(j);
					//	System.out.println("     OK");					
					}
					//else
						//System.out.println("     NO");
				}
				
				if(instDistVal.size()!=0)
				{
					//Extraires les exemples associ�s a cet attribut
					ArrayList <Integer> FERM_exe = new ArrayList <Integer> ();	
					ArrayList <String> FERM_att = new ArrayList <String> ();	
			
					//Liste des instances verifiant la fermeture
					for(int i=0; i<inst.numInstances(); i++)
					{
						if( inst.instance(i).stringValue(m_Attribute.index()) == m_Attribute.value(indDistVal))
							FERM_exe.add(i);
					}
					int nbrInstFer = FERM_exe.size();
					
					if(m_Debug)
					{	
						System.out.print("\nFermeture d'instances:\n\tTaille : "+ nbrInstFer+"\n\tLes indices : ");
						for(int i=0; i<nbrInstFer; i++)
							System.out.print (" - "+FERM_exe.get(i));
						System.out.println();
						for(int i=0; i<nbrInstFer; i++)
							System.out.println(FERM_exe.get(i)+" : "+inst.instance(FERM_exe.get(i)).toString());
					}
					
					//Liste des attributs associ�s � la fermeture ??????????????????????
					//System.out.println("Extraction des attributs associ�s � cette fermeture");
					String nl= "-null-";
					for(int i=0; i< (int) inst.numAttributes()-1;i++)
					{
						int cmpt=0;
						String FirstDistVal = inst.instance(FERM_exe.get(0)).stringValue(i);
						//System.out.println("Attribut d'indice: "+i+" FirstDistVal: "+FirstDistVal);
						for (int j=0; j<nbrInstFer;)
						{
							if(inst.instance(FERM_exe.get(j)).stringValue(i)== FirstDistVal) 
							{
								cmpt++;
								if(cmpt==nbrInstFer)
								{
									FERM_att.add(inst.instance(FERM_exe.get(0)).stringValue(i));
									//System.out.println(" ---> ok ");
								}
								j++;
							}
							else
							{
								j=nbrInstFer;
								FERM_att.add(nl); 
								//System.out.println(" ---> null ");
							}
						}
					}
					
					if(m_Debug)
					{
					System.out.println ("\nFermeture d'attributs est de taille : "+ FERM_att.size()+" : ");
					for(int i=0; i<FERM_att.size(); i++)
						System.out.print (FERM_att.get(i)+" , ");
					}
					
					
					//Extraire la classe majoritaire associ�e//		
					int [] nbClasse = new int[inst.numClasses()];			
					for(int k=0;k<inst.numClasses();k++)
						nbClasse[k]=0;
						
					//Parcourir les exemples associ�e � ce concept
					//System.out.println();
					for(int j=0;j<nbrInstFer;j++)
						nbClasse[(int)inst.instance(FERM_exe.get(j)).classValue()]++;
						
						
					//Detertminer l'indice de la classe associ�
					int indiceMax=0;
					for(int i=0;i<inst.numClasses();i++)
						if(nbClasse[i]>nbClasse[indiceMax])
							indiceMax=i;
					
					if(m_Debug)	{
						System.out.println ("\nLa Classe Majoritaire est d'indice: "+ indiceMax+" : "+inst.attribute(inst.classIndex()).value(indiceMax));
						System.out.println ("Liste des des attribut de la fermeture");
						for (int o=0;o<FERM_att.size();o++)
							System.out.print (FERM_att.get(o)+" , ");
						System.out.println ("");
					}
							
					//On retourne le concept Pertinent comme un vecteur 
					ArrayList <String> CP= new ArrayList <String>();
					for (int i=0; i< (int) inst.numAttributes()-1;i++)
							CP.add(FERM_att.get(i));
					
					Classification_Rule r = new Classification_Rule(CP,indiceMax,inst.attribute(inst.classIndex()).value(indiceMax));
					
					classifierNC.add(r);
					}
			}
			
	    }
	    */

	return classifierNC;
	}
}
