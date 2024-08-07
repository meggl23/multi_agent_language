This jupyter notebook connects to our GitHub repo and downloads the necessary data from there. Before running it please make sure to install the requirements into your environment and load the data as follows:
- `git clone https://github.com/meggl23/multi_agent_language --branch v3.0`
- `cd multi_agent_language`
- `unzip data.zip`

Now, at the bottom of the first cell specify the location of the newly created data folder
- `"/your/working/dir/.../multi_agent_language/data/"`
The notebook is now ready to run! For any figure that you want to create, always execute the first 19 notebook cells before executing a "PLOT" cell. After cell 1 has been successfully executed once, you can uncomment the second line to skip the imports.

The following guidelines show how to generate each of the plots in order of their appearance in the paper

Figs. 2a, S5a, S6a, S7a, S8a:
- in cell 2, set language_code="nonlinear_nostudent_language0" 
- execute the first 19 cells
- execute cell 20

Figs. 2b,c, S5b,c, S6b,c, S7b,c, S8b,c:
- in cell 2, set language_code="nonlinear_goallocs0_zeta5_language2"
- execute the first 19 cells
- execute cell 20

Figs. 3b,c:
- execute the first 19 cells
- execute cells 24 and 25

Fig. 3e:
- execute the first 19 cells
- execute cells 26 and 27

Figs. 4a-d:
- in cell 2, set zeta_lossplot=5
- execute the first 19 cells
- execute cell 21

Figs. 4e-h:
- in cell 2, set folders_goalloc_plots=[f"nonlinear_goallocs{i}_zeta5_factor1" for i in range(7)]
- in cell 2, set language_nr_goalloc_plots=5
- in cell 2, set chuck_out=False
- execute the first 19 cells
- execute cell 22

Figs. 5a, S10:
- in cell 2, set language_codes_closingloop=["nonlinear_goallocs0_zeta5_language0"]
- execute the first 19 cells
- execute cell 23

Figs. 5b-e:
- in cell 2, set folders_goalloc_plots=[f"studentQfeedback_goallocs{i}_zeta5_factor1" for i in range(7)]
- in cell 2, set language_nr_goalloc_plots=25
- in cell 2, set chuck_out=True
- execute the first 19 cells
- execute cell 22

Figs. S1a:
- in cell 2, set language_code="linear_ae_nostudent_language0"
- execute the first 19 cells
- execute cell 20

Figs. S1b,c:
- in cell 2, set language_code="linear_both_zeta5_language0"
- execute the first 19 cells
- execute cell 20

Figs. S2:
- in cell 2, set language_code="linear_ae_zeta5_language0"
- execute the first 19 cells
- execute cell 20

Figs. S3:
- in cell 2, set folders_goalloc_plots=[f"nonlinear_frozen_goallocs{i}_zeta5_factor1" for i in range(7)]
- in cell 2, set language_nr_goalloc_plots=5
- in cell 2, set chuck_out=False
- execute the first 19 cells
- execute cell 22

Figs. S4:
- in cell 2, set folders_goalloc_plots=[f"nonlinear_frozen2_goallocs{i}_zeta5_factor1" for i in range(7)]
- in cell 2, set language_nr_goalloc_plots=5
- in cell 2, set chuck_out=False
- execute the first 19 cells
- execute cell 22

Figs. S9:
- execute the first 19 cells
- execute cell 30

Figs. S11:
- in cell 2, set zeta_lossplot=1 (a), 2 (b), 5 (c), 10 (d)
- execute the first 19 cells
- execute cell 21

Figs. S12:
- execute the first 19 cells
- execute cells 28 and 29

Figs. S13:
- in cell 2, set folders_goalloc_plots=[f"studentQfeedback_goallocs{i}_zeta5_factor1" for i in range(7)]
- in cell 2, set language_nr_goalloc_plots=25
- in cell 2, set chuck_out=False
- execute the first 19 cells
- execute cell 22




In the following, we describe the most important training and evaluation procedures to test and experiment with the framework:




How to train a language:

- in cell 3, set language_gen=True
- in cell 3, set other parameters of the language like message length (K), loss parameters gamma_sparse, kappa, zeta_std, learning rate, number of training epochs
- execute the first 17 cells
- in cell 18, tune more language parameters like which parameters should be frozen, which tasks should be trained on, nonlinearities, and the name of your language or languages (language_save_code)
- execute cell 18
(the tag "_language0" will automatically be appended to your trained language. If you train more than 1 language with the same parameters but different seeds, the following languages will have the tags "_language1", "_language2",... appended) 
- afterwards, set language_gen=False again


How to show message plots of a trained language:

- in cell 2, set language_code=[your_language_name], which was set during training as the parameter language_save_code (plus a tag like "_language0")
- if the plots should be saved, set save_message_plots=True in cell 2
- execute the first 19 cells
- execute cell 20



How to evaluate student performances of a trained language:

- in cell 3, set student_evaluate=True (and set save_rates=True if the solving rates should be saved)
- in cell 2, set qmat_read_code="training4x4" if you are interested in training matrices and set qmat_read_code="test4x4" if you are interested in test messages
- execute the first 18 cells
- in cell 19, set language_codes=[your_language_name] (NOT including tags like "_language0"!) and set language_nr_evaluation to the number of languages you trained with identical parameters and names, but different seeds
- execute cell 19
- afterwards, set student_evaluate=False again



How to get the student messages for closing the loop:

- in cell 2, set language_codes_closingloop=[your_language_names]
- in cell 2, set qmat_read_code="training4x4" if you are interested in training matrices and set qmat_read_code="test4x4" if you are interested in test messages
- execute the first 19 cells
- execute cell 23 --> the student Q-matrices are generated and saved under "/closing the loop/[your_language_name]/studentQmatrices.pkl
- now follow the steps for evaluating the student performance of a trained language and make sure to set q_matrix_dict_list=[read_dict_from_pkl([your_student_Qmatrix_file_location])]






In the following, we list the names of the languages used for evaluations in the paper and their corresponding parameters:
The following are all language codes of languages that can be analyzed:

#languages without feedback
- nonlinear_nostudent_language{x} x=0,1,2,3,4

#languages with feedback, zeta=5, training on different goal locations
- nonlinear_goallocs{x}_zeta5_language{y} x=0,1,2,3,4,5,6 / y=0,1,2,3,4,5,...,23,24

#hyperparameter analysis: number of trained epochs
- nonlinear_epochs{x}_language{y} x=100,200,400,600,800,1000 / y=0,1,2,3,4
#hyperparameter analysis: zeta
- nonlinear_zeta{x}_language{y} x=1,2,5,10,20 / y=0,1,2,3,4
#hyperparameter analysis: message length
- nonlinear_mlength_K{x}_language{y} x=1,2,3,4,5,6,7,8 / y=0,1,2,3,4

#linearity analysis: linear autoencoder, no student
- linear_ae_nostudent_language{x} x=0,1,2,3,4 
#linearity analysis: linear autoencoder, nonlinear student
- linear_ae_zeta5_language{x} x=0,1,2,3,4
#linearity analysis: linear autoencoder, linear student
- linear_both_zeta5_language{x} x=0,1,2,3,4

#training on frozen messages created with feedback
- nonlinear_frozen_feedback_goallocs{x}_zeta5_language{y} x=0,1,2,3,4,5,6 / y=0,1,2,3,4
#training on frozen messages created without feedback
- nonlinear_frozen_nofeedback_goallocs{x}_zeta5_language{y} x=0,1,2,3,4,5,6 / y=0,1,2,3,4

#regularization loss set to zero (entropy analysis)
- nonlinear_noregularization_language{x} x=0,1,2,3,4









