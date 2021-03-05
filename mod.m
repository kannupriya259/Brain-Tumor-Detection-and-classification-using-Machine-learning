%%% BRAIN TUMOUR  DETECTION AND CLASSIFICATION)%%% 

clc;

close all;

clear all;

warning('off','all');

%%READ INPUT%%%

[FileName,PathName] = uigetfile('*.jpg;*.png;*.bmp','Pick an MRI Image');
if isequal(FileName,0)||isequal(PathName,0)
    warndlg('User Press Cancel');
else
    
a = imread([PathName,FileName]);
figure,imshow(a),title('INPUT IMAGE');

%%%%%%% RESIZE INPUT row column%%%%%%
a=imresize(a,[256 256]);
figure,imshow(a),title('RESIZED INPUT IMAGE');

%%%%%%CONVERT RESIZED INPUT INTO GRAYSCALE roi segm%%%%%%
[m n o]=size(a);
if o==3
    gray=rgb2gray(a);
else 
    gray=a;
end
end
 figure,imshow(gray),title('GRAY SCALE IMAGE');


%%%%%%%REGION BASED SEGMENTATION%%%%%%%
   
%%%%%ADJUSt GRAY IMAGE%%%%%
bw1=imadjust(gray);
figure,imshow(bw1),title('CONTRAST ENHANCED GRAYSCALE IMAGE');
    
%%CONVERT TO BLACK AND WHITE%%%
bw=im2bw(bw1,0.7);

figure,imshow(bw),title('BLACK AND WHITE IMAGE');

    
% %%LABEL BLACK AND WHITE sobel %%%
label=bwlabel(bw);

figure,imshow(label),title('LABELED IMAGE');
 
    %%%%SELECT THE REGION PROPERTY%%%%
      %%%%%TO CALL%%%%%
stats=regionprops(label,'all');
     
%%%%%IMPORTANT PROPERTIES%%%%%%
area=[stats.Area];
centroid=[stats.Centroid];
majorAxisLength=[stats.MajorAxisLength];
minorAxisLength=[stats.MinorAxisLength];
eccentricity=[stats.Eccentricity];
orientation=[stats.Orientation];
filledArea=[stats.FilledArea];
equivdiameter=[stats.EquivDiameter];
density=[stats.Solidity];
perimeter=[stats.Perimeter];

%%%%%%FROM that 'AREA' AND 'DENSITY' %%%%%%
 
%%%%HIGH DENSE AREA%%%%
high_dense_area=density>0.3;
  
%%%%MAX AREA WITH HIGH DENSITY%%%%
max_area=max(area(high_dense_area));
 
%%%%FIND THE MAXIMUM AREA%%%%
tumour_label=find(area==max_area);
  
%%%%LABEL THAT THE MAXIMUM AREA%%%%
tumour=ismember(label,tumour_label);

%%%%%%%DILATE  THE TUMOR AREA morphological%%%%%%
se=strel('square',8);
tumour=imdilate(tumour,se);

 
%%%%TO SHOW THE TUMOR AREA ALONE%%%%%
figure,imshow(tumour,[]),title('TUMOR ALONE IMAGE');

 %%%%%TO SHOW THE DETECTED TUMOR IN THE INPUT IMAG boundary tracing%%%%
[B,L]=bwboundaries(tumour,'noholes');
figure,imshow(a,[]);
hold on
for  i=1:length(B)
    plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
end
title('DETECTED TUMOUR IMAGE')
hold off



%%%TO PLOT THE INPUT IMAGE,TUMOUR ALONE IMAGE ,AND DETECTED TUMOR ON INPUT IMAGE%%%
figure,
subplot(1,3,1),imshow(a,[]),title('INPUT TUMOR IMAGE');
subplot(1,3,2),imshow(tumour,[]),title('TUMOUR ALONE IMAGE');
[B,L]=bwboundaries(tumour,'noholes');
subplot(1,3,3),imshow(a,[]);
hold on
for  i=1:length(B)
    plot(B{i}(:,2),B{i}(:,1),'g','linewidth',1.5)
end
title('TUMOUR DETECTED')
hold off
set(gcf, 'Position', get(0,'Screensize'));


%%%%FEATURE EXTRACTION BY PCA AND GLCM%%%%%

  %%FEATURE EXTRACTION BY PCA(PRINCIPAL COMPONENT ANALYSIS large to small)%%%

 signal1=tumour(:,:);
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');
DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);


   %%%%%FEATURE EXTRACTION BY GLCM(GRAY LEVEL CO-OCCURENCE MATRIX neighbor horzntly scale)%%%%%
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
Skewness = skewness(G);
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));

%%% Inverse Difference Movement(homogeneity)%%%
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
feat_disease1 = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];


%%%%%%% Color Texture Feature Extraction(LBP) FOR TUMOR ALONE IMAGE%%%
%%%%% step 1: Local Binary Patterns 
feat_LBP = extractLBPFeatures(tumour);
grayImage = tumour;
localBinaryPatternImage1 = zeros(size(grayImage));
[row col] = size(grayImage);
for r = 2 : row - 1   
	for c = 2 : col - 1    
		centerPixel = grayImage(r, c);
		pixel7 = grayImage(r-1, c-1) > centerPixel;  
		pixel6 = grayImage(r-1, c) > centerPixel;   
		pixel5 = grayImage(r-1, c+1) > centerPixel;  
		pixel4 = grayImage(r, c+1) > centerPixel;     
		pixel3 = grayImage(r+1, c+1) > centerPixel;    
		pixel2 = grayImage(r+1, c) > centerPixel;      
		pixel1 = grayImage(r+1, c-1) > centerPixel;     
		pixel0 = grayImage(r, c-1) > centerPixel;       
		localBinaryPatternImage1(r, c) = uint8(...
			pixel7 * 2^7 + pixel6 * 2^6 + ...
			pixel5 * 2^5 + pixel4 * 2^4 + ...
			pixel3 * 2^3 + pixel2 * 2^2 + ...
			pixel1 * 2 + pixel0);
	end  
end 

% figure,imshow(localBinaryPatternImage1),title('LBP TUMOR-ALONE IMAGE'); 



%%%%%% COMBINE THE TWO FEATURES %%%%%%%

feat_disease=[feat_disease1 feat_LBP];


%%%%LOAD ALL THE FEATURES%%%%
load featurebrain1.mat



 
 
%%%%%SVM CLASSIFICATION%%%%%

test1 = zeros(1,4);
test1(1:2)= 1;
test1(3:4)=2;
A=svmtrain(feature_final1,test1);
result=svmclassify(A,feat_disease);

if result==1
    
    stats=regionprops(tumour,'all');
    area=[stats.Area];
    
    if(area<1000)
        
    msgbox('BENIGN TUMOR');
    warndlg('normal stage');
    disp('BENIGN TUMOR-normal stage');
    
    elseif(area>=1000&&area<2000)
        
        msgbox('BENIGN TUMOR');
         warndlg('medium stage');
         disp('BENIGN TUMOR-medium stage');
    
    elseif(area>=2000)
        msgbox('BENIGN TUMOR');
        warndlg('severe stage');
        disp('BENIGN TUMOR- stage');
        
    end
    
elseif result==2
    
    stats=regionprops(tumour,'all');
    area=[stats.Area];
    
    if(area<5000)
        
    msgbox('MALIGNANT TUMOR');
    warndlg('normal stage');
    disp('MALIGNANT TUMOR-normal stage');
    
    elseif(area>=5000&&area<10000)
        
        msgbox('MALIGNANT TUMOR');
         warndlg('medium stage');
        disp('MALIGNANT TUMOR-medium stage');
        
    elseif(area>=10000)
        msgbox('MALIGNANT TUMOR');
        warndlg('severe stage');
    disp('MALIGNANT TUMOR- stage');
    
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




