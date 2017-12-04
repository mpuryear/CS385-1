function do_svm_evaluation(config_file)

%% Test and plot graphs for a svm classifier learnt with do_svm.m

%% The action of this routine depends on the directory in which it is
%% run: 
%% (a) If run from RUN_DIR, then it will evaluate the latest model in the
%% models subdirectory. i.e. if you have just run
%% do_plsa('config_file_2'), which saved to model_0011.mat and
%% config_file_0011.m in the models subdirectory in RUN_DIR, then doing
%% do_plsa_evaluation('config_file_2') will load up model_0011.mat and
%% evaluate it. 
%% (b) If run within in models subdirectory, then it
%% will evaluate the model corresponding to the configuration file passed
%% to it. i.e. do_plsa_evaluation('config_file_0002') will load
%% model_0002.mat and evaluate/plot figures for it. 
%%  
%% Mode (a) exists to allow a complete experiment to be run from start to
%% finish without having to manually go into the models subdirectory and
%% find the appropriate one to evaluate.
  
%% If this routine is called on a newly learnt model, it will run the pLSA code
%% in folding in mode and then plot lots of figures. If run a second time
%% on the same model, it will only plot the figures, since there is no need
%% to recompute the statistics on the testing images. If you want to force it
%% to re-run on the images, then remove the Pc_d_pos_test variable from the
%% model file. 
  
%% Note this only uses a pre-existing model to evaluate the test
%% images. Please use do_naive_bayes to actually learn the classifiers.  
%% Before running this, you must have run:
%%    do_random_indices - to generate random_indices.mat file.
%%    do_preprocessing - to get the images that the operator will run on.  
%%    do_interest_op  - to get extract interest points (x,y,scale) from each image.
%%    do_representation - to get appearance descriptors of the regions.  
%%    do_vq - vector quantize appearance of the regions in each image.
%%    do_naive_bayes - learn a Naive Bayes classifier.
  
%% R.Fergus (fergus@csail.mit.edu) 03/10/05.  

%% figure numbers to start at
FIGURE_BASE = 2000;
%% color ordering
cols = {'g' 'r' 'b' 'c' 'm' 'y' 'k'};
markers = {'+', '.'};

%% Evaluate global configuration file
eval(config_file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Model section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% get filename of model to learn
%%% if in models subdirectory then just get index off config_file string
if (strcmp(pwd,[RUN_DIR,'/',Global.Model_Dir_Name]) | strcmp(pwd,[RUN_DIR,'\',Global.Model_Dir_Name]))
    ind = str2num(config_file(end-Global.Num_Zeros+1:end));
else
    %%% otherwise just take newest model in subdir.
    ind = length(dir([RUN_DIR,'/',Global.Model_Dir_Name,'/',Global.Model_File_Name,'*.mat']));    
end
%%% construct model file name
model_fname = [RUN_DIR,'/',Global.Model_Dir_Name,'/',Global.Model_File_Name,prefZeros(ind,Global.Num_Zeros),'.mat'];

%%% load up model
load(model_fname);

%% get +ve interest point file names
pos_ip_file_names = [];
pos_sets = find(Categories.Labels==1);
for a=1:length(pos_sets)
    pos_ip_file_names =  [pos_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Test_Frames{pos_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end

%% get -ve interest point file names
neg_ip_file_names = [];
neg_sets = find(Categories.Labels==0);
for a=1:length(neg_sets)
    neg_ip_file_names =  [neg_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Test_Frames{neg_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test section - run model on testing images only if Pd_z_test does not exist
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if ~exist('Pc_d_pos_test') %%% only do this section the first time we look at the model
                       %%% saves time if we just want to look at the pretty
                       %%% figures

%% Create matrix to hold word histograms from +ve images
X_fg = zeros(VQ.Codebook_Size,length(pos_ip_file_names));

%% load up all interest_point files which should have the histogram
%% variable already computed (performed by do_vq routine).
for a=1:length(pos_ip_file_names)
    %% load file
    load(pos_ip_file_names{a});
    %% store histogram
    X_fg(:,a) = histg';    
end 


%% Create matrix to hold word histograms from +ve images
X_bg = zeros(VQ.Codebook_Size,length(neg_ip_file_names));

%% load up all interest_point files which should have the histogram
%% variable already computed (performed by do_vq routine).
for a=1:length(neg_ip_file_names)
    %% load file
    load(neg_ip_file_names{a});
    %% store histogram
    X_bg(:,a) = histg';    
end 

%% concatinate both of our test sets and transpose them to fit into our predict
testX = cat(2, X_fg, X_bg);
testX = testX';


%%% Compute ROC and RPC on test data
labels = [ones(1,length(pos_ip_file_names)) , zeros(1,length(neg_ip_file_names))];

%%% predict our values and transpose to match our labels
[~, values] = predict(SVMModel, testX);
values = values(:,1)';


%%% compute roc
[roc_curve_test,roc_op_test,roc_area_test,roc_threshold_test] = roc([values;labels]');
fprintf('Testing: Area under ROC curve = %f\n', roc_area_test);
%%% compute rpc
[rpc_curve_test,rpc_ap_test,rpc_area_test,rpc_threshold_test] = recall_precision_curve([values;labels]',length(pos_ip_file_names));
fprintf('Testing: Area under RPC curve = %f\n', rpc_area_test);

%%% save variables to file
save(model_fname,'roc_curve_test','roc_op_test','roc_area_test','roc_threshold_test','rpc_curve_test','rpc_ap_test','rpc_area_test','rpc_threshold_test','-append');
    
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting section - plot some figures to see what is going on...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% We will use figures from FIGURE_BASE to FIGURE_BASE + 4;
%% clear them ready for plotting action...
for a=FIGURE_BASE:FIGURE_BASE+4
    figure(a); clf;
end

%% Now lets look at the classification performance
figure(FIGURE_BASE); hold on;
plot(roc_curve_train(:,1),roc_curve_train(:,2),'r');
plot(roc_curve_test(:,1),roc_curve_test(:,2),'g');
axis([0 1 0 1]); axis square; grid on;
xlabel('P_{fa}'); ylabel('P_d'); title('ROC Curves');
legend('Train','Test');

%% Now lets look at the retrieval performance
figure(FIGURE_BASE+1); hold on;
plot(rpc_curve_train(:,1),rpc_curve_train(:,2),'r');
plot(rpc_curve_test(:,1),rpc_curve_test(:,2),'g');
axis([0 1 0 1]); axis square; grid on;
xlabel('Recall'); ylabel('Precision'); title('RPC Curves');
legend('Train','Test');





   


        
