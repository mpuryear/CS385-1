function DoG(image_file_names,output_file_names)

  %% Simple interest operator that:
  %%    1. Runs difference of Gaussian operator on image using VL FEAT package.

  %% Inputs: 
  %%      1. image_file_names - cell array of filenames of all images to be processed
  %%      2. output_file_names - cell array of output filenames

  %% Outputs:
  %%      None - it saves the results for each image to the files
  %%      specified in output_file_names.
  %%      Each file holds 4 variables:
  %%          x - x coordinates of points (1 x NumPoints)
  %%          y - y coordinates of points (1 x NumPoints)
  %%          scale - characteristic scale of points (radius, in pixels)  (1 x NumPoints)
  %%          angle - dominant angle of the region around each keypoint  (1 x NumPoints)
  %%          score - Always 1 (1 x NumPoints).
  %%          descriptor - SIFT descriptors of all keypoints (128 x NumPoints)

    
%%% Get total number of images
nImages = length(image_file_names);

%%% Loop over all images
for i = 1:nImages


  % read in ith image
  ith_img = imread(image_file_names{i});
     
  % Convert to single, grayscale image
  gray = ith_img;
  
  % only gray our image if its not already gray
  if size(gray,3) == 3
    gray = rgb2gray(gray);
  end
  
  ith_gray = single(gray);
  
  % Find Interest points using vl_sift. Store features in f and descriptors in d.
  [f,d] = vl_sift(ith_gray);

  % Total number of features from image
  nFeats = size(f,2);
  
  % From f and d extract x, y, scale, angle and descriptor.
  x = f(1,:);
  y = f(2,:);
  scale = f(3,:);
  angle = f(4,:);
  descriptor = d;
  
  %%
  score = ones(1, nFeats);


  fprintf('Image: %d, Number of features detected: %d\n',i,length(x));
  % Save in output file
  
  save(output_file_names{i}, 'x', 'y','scale','score','angle','descriptor');
  
   %%% print out progress every 10 images    
   if (mod(i,10)==0)
      fprintf('%d.',i);
   end
end
