function do_svm(config_file)


%% Note that this only trains a model. It does not evaluate any test
%% images. Use do_svm_evaluation for that.  
  
%% Before running this, you must have run:
%%    do_random_indices - to generate random_indices.mat file
%%    do_preprocessing - to get the images that the operator will run on  
%%    do_interest_op  - to get extract interest points (x,y,scale) from each image
%%    do_representation - to get appearance descriptors of the regions  
%%    do_vq - vector quantize appearance of the regions
  
%% R.Fergus (fergus@csail.mit.edu) 03/10/05.  
 
    
%% Evaluate global configu
%% Evaluate global configuration file
eval(config_file);

%% ensure models subdir is present
[s,m1,m2]=mkdir(RUN_DIR,Global.Model_Dir_Name);

%% get all file names of training image interest point files

%% get +ve interest point file names
pos_ip_file_names = [];
pos_sets = find(Categories.Labels==1);
for a=1:length(pos_sets)
    pos_ip_file_names =  [pos_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Train_Frames{pos_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end

%% get -ve interest point file names
neg_ip_file_names = [];
neg_sets = find(Categories.Labels==0);
for a=1:length(neg_sets)
    neg_ip_file_names =  [neg_ip_file_names , genFileNames({Global.Interest_Dir_Name},Categories.Train_Frames{neg_sets(a)},RUN_DIR,Global.Interest_File_Name,'.mat',Global.Num_Zeros)];
end

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


%% Create matrix to hold word histograms from -ve images
X_bg = zeros(VQ.Codebook_Size,length(neg_ip_file_names));

%% load up all interest_point files which should have the histogram
%% variable already computed (performed by do_vq routine).
for a=1:length(neg_ip_file_names)
    %% load file
    load(neg_ip_file_names{a});
    %% store histogram
    X_bg(:,a) = histg';    
end 


%% concatinate our two sets of images.
trainX = cat(2, X_fg, X_bg);

%% create all the labels for our images. first 50% are 1 second 50% are 0
trainY = zeros(length(pos_ip_file_names) + length(pos_ip_file_names), 1);

for a=1:length(pos_ip_file_names)
   trainY(a) = 1;
end

for a=length(pos_ip_file_names) + 1:length(pos_ip_file_names) + length(pos_ip_file_names)
   trainY(a) = 0;
end

%% we must make X 100x300  as N is training Points and d is CodeWords.
%% thus Nxd instead of dxN
trainX = trainX';

%% this combination of linear/nonlinear kernals and standardization yielded the best/correct results
SVMModel = fitcsvm(trainX, trainY, 'KernelFunction','rbf','Standardize',true, 'KernelScale', 'auto', 'ClassNames', [1, 0]);


%%% Compute ROC and RPC on training data
labels = [ones(1,length(pos_ip_file_names)) , zeros(1,length(neg_ip_file_names))];

%%% predict our values
[~, values] = predict(SVMModel, trainX);

% translate our values to fit our roc curve
values = values(:,1)'; 

%%% compute roc
[roc_curve_train,roc_op_train,roc_area_train,roc_threshold_train] = roc([values; labels]');
fprintf('Training: Area under ROC curve = %f; Optimal threshold = %f\n', roc_area_train, roc_threshold_train);
%%% compute rpc
[rpc_curve_train,rpc_ap_train,rpc_area_train,rpc_threshold_train] = recall_precision_curve([values;labels]',length(pos_ip_file_names));
fprintf('Training: Area under RPC curve = %f\n', rpc_area_train);
%%% Now save model out to file
[fname,model_ind] = get_new_model_name([RUN_DIR,'/',Global.Model_Dir_Name],Global.Num_Zeros);

%%% save variables to file
%% save our SVMModel to be used by our evaluation
save(fname,'SVMModel','roc_curve_train','roc_op_train','roc_area_train','roc_threshold_train','rpc_curve_train','rpc_ap_train','rpc_area_train','rpc_threshold_train');

%%% copy conf_file into models directory too..
config_fname = which(config_file);
copyfile(config_fname,[RUN_DIR,'/',Global.Model_Dir_Name,'/',Global.Config_File_Name,prefZeros(model_ind,Global.Num_Zeros),'.m']);
