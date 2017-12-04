function driver()


config_file = 'config_file_1';
%% copy all of the images in our folder for restoration between different preprocessing events
path_with = 'Final/images/with/';
path_without = 'Final/images/without/';
files_with = dir2(strcat(path_with, '**'));
files_without = dir2(strcat(path_without, '**'));


%% read in all the images and store/index them
for k = 1:numel(files_with)
  filename_with{k}   = files_with(k).name;
  img_originals_with{k}     = imread(strcat(path_with,filename_with{k}));
  img_copies_with{k} = img_originals_with{k};
  [~,name_with{k},~] = fileparts(filename_with{k});
end

for k = 1:numel(files_without)
  filename_without{k}   = files_without(k).name;
  img_originals_without{k}     = imread(strcat(path_without, filename_without{k}));
  img_copies_without{k} = img_originals_without{k};
  [~,name_without{k},~] = fileparts(filename_without{k});
end



%% no custom preprocessing

%%% run configuration script robustly to get EXPERIMENT_TYPE
%%% this tells if we are doing plsa or a bag or words or a parts and
%%% structure experiment etc.
try
    eval(config_file);
catch
    
end

%% call of these functions, except do_svm/naive, prior to testing new images.

  %  pre_test(config_file);
    
  %  test(config_file, 'no filter');


%{
%% check results after image processing
    
    % display unmodified image
    figure, imshow(img_copies_with{1})      

  % Modify and store copied images in place of our original images.
   for k = 1:numel(files_with)
       
     % gray any images that are color before canny edge
    if size(img_copies_with{k},3) == 3
        img_copies_with{k} = rgb2gray(img_copies_with{k});
    end
    
    img_copies_with{k} = edge(img_copies_with{k}, 'canny');
    
    imwrite(img_copies_with{k},strcat(path_with, filename_with{k}));
   end
   
   
   for k = 1:numel(files_without)
             
     % gray any images that are color before canny edge
    if size(img_copies_without{k},3) == 3
        img_copies_without{k} = rgb2gray(img_copies_without{k});
    end
    
    img_copies_without{k} = edge(img_copies_without{k}, 'canny');
    
    imwrite(img_copies_without{k},strcat(path_without, filename_without{k}));
   end
 
   %% display the now modified image
   figure, imshow(img_copies_with{1})
   
   pre_test(config_file);
   
   test(config_file, 'canny');
   
    % restore our state back to before this test 
   img_copies_with = img_originals_with;  
   img_copies_without = img_originals_without;   
  
 %}
  
%% Total Variation Denoising

lambda = 1.0;
niter = 20;
se = strel('disk', 5);

  % Modify and store copied images in place of our original images.
   for k = 1:numel(files_with)
       
     % gray any images that are color before canny edge
    if size(img_copies_with{k},3) == 3
        img_copies_with{k} = rgb2gray(img_copies_with{k});
    end
    
    %img_copies_with{k} = TVL1denoise(img_copies_with{k}, lambda, niter);
   
    img_copies_with{k} = imdilate(img_copies_with{k}, se);
    img_copies_with{k} = imerode(img_copies_with{k}, se);
     
    fprintf('part1 %d / %d complete\n', k, numel(files_with))
    
    imwrite(img_copies_with{k},strcat(path_with, filename_with{k}));
   end
   
   
   for k = 1:numel(files_without)
             
     % gray any images that are color before canny edge
    if size(img_copies_without{k},3) == 3
        img_copies_without{k} = rgb2gray(img_copies_without{k});
    end
    
    %img_copies_without{k} = TVL1denoise(img_copies_without{k}, lambda, niter);
    img_copies_without{k} = imdilate(img_copies_without{k}, se);
    img_copies_without{k} = imerode(img_copies_without{k}, se);
    
    fprintf('part2 %d / %d complete\n', k, numel(files_with))
    
    imwrite(img_copies_without{k},strcat(path_without, filename_without{k}));
   end

% test

   %% display the now modified image
   figure, imshow(img_copies_with{1})
   
   pre_test(config_file);
   
   test(config_file, 'total variation denoising');

 %{
%% check results after image processing
do_all('config_file_1')

%}
   
%restore images to their original state
   for k = 1:numel(files_with)
    imwrite(img_originals_with{k},strcat(path_with, filename_with{k}));
   end
   
   for k = 1:numel(files_without)
       imwrite(img_originals_without{k}, strcat(path_without, filename_without{k}));
   end

end


function test(config_file, filter_name) 
    fprintf('\nNaive bayes using %s\n', filter_name);
    do_naive_bayes(config_file); 
    do_naive_bayes_evaluation(config_file);
    
    fprintf('\nSVM using %s\n', filter_name);
    do_svm(config_file);    
    do_svm_evaluation(config_file);
end

function pre_test(config_file) 
    %%% generate random indices for trainig and test frames
    do_random_indices(config_file);
    
    %%% copy & resize images into experiment subdir
    do_preprocessing(config_file);
    
    %%% run interest operator over images and obtain representation of interest points
    do_interest_operator(config_file);

    %%% form appearance codebook
    do_form_codebook(config_file);
    
    %%% VQ appearance of regions
    do_vq(config_file);

end


function listing = dir2(varargin)

if nargin == 0
    name = '.';
elseif nargin == 1
    name = varargin{1};
else
    error('Too many input arguments.')
end

listing = dir(name);

inds = [];
n    = 0;
k    = 1;

while n < 2 && k <= length(listing)
    if any(strcmp(listing(k).name, {'.', '..'}))
        inds(end + 1) = k;
        n = n + 1;
    end
    k = k + 1;
end

listing(inds) = [];
end