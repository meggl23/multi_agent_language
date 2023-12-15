- The notebook can be run out of the box, as it connects to our GitHub repo and downloads the necessary data from there.
- In particular, the first notebook cell is responsible for the imports and cloning of the repo

- For any figure that you want to create, ALWAYS execute the first 20 notebook cells before executing a "PLOT" cell
- After this has been done once, uncomment the second line in cell 1
- --> this will save time in subsequent runs, as the imports don't need to be done multiple times

The following guidelines show how to generate each of the plots in order of their appearance in the paper

Figs. 2 and S9-12 (together): 
- in cell 3, enter language_code="nonlinear_goallocs0_zeta5_language2"
- execute cells 1-20
- execute cell 21

Fig. 3a:
- 




Most figures should now be visible (look at the individual notebook cells, they describe what figures are created there) 
In particular, to get message plots for different languages, the parameter "language_code" has to be changed. The notebook cell "Guideline/dictionary for the changeable parameters" lists more possible languages to explore. Change the parameter "language_code" in the cell "Changeable parameters" to the language of interest and rerun the entire notebook.

For training a new language yourself, please change the parameters in the notebook cell "INIT: fixed parameters" under the point "autoencoder network parameters for training the language" accordingly. Importantly, the parameter "language_gen" needs to be set to True and the value of "language_number_autoenc" determines how many languages should be trained (all with the specified parameters, but different seeds). 

When this is done, go to the cell "Execution: language training". Here you need to specify the names of the new language(s) in the string "language_save_code". The parameter "train_goals_index" now specifies, which goal locations should be included in the training (see dictionary "train_goals_dict" in the same notebook cell). Now you are ready to start the training! Afterwards, you can enter the name of your new language in the parameter "language_code", and run the notebook again (WHILE NOW NOT TRAINING A LANGUAGE, i.e. THE PARAMETER "LANGUAGE_GEN" NEEDS TO BE SET TO FALSE) to see the message space plots among others. Note, that if the number of languages you trained at the same time is larger than 1, your language names will be "[your_language_name]_language{i}" with i=0,1,2,...
